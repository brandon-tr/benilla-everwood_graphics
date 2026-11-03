//! Camera-only reprojection candidates for stationary opaque surfaces.
//!
//! This lane builds on the RP-02A depth publication and the current/previous depth and mask pairs.
//! It produces a per-view `Rgba16Float` candidate texture containing current-minus-previous UV
//! motion, expected previous reverse depth and a fused-valid flag. The flag is set only when the
//! geometric reprojection, both current masks, both previous masks across the clamped 2x2 previous
//! footprint, and all four previous-depth comparisons pass. It is **not** object/deformation
//! velocity, a rigid-surface classification, confidence, complete scene coverage, or final reuse
//! authorization. The gate is evaluated at full precision before RGBA16F storage; a later resolve
//! must account for that storage precision and its own sampling footprint.
//!
//! The camera baseline advances only in [`RenderSystems::Cleanup`], after both this lane's node and
//! the depth node have recorded their commands. Preparation never advances it. Missing history or
//! pipeline/uniform readiness still encodes a cleared zero-invalid candidate so a first frame can
//! seed the next camera baseline; missing current depth skips encoding and resets that baseline.
//!
//! This supports ordinary full-target cameras, including bounded Bevy perspective jitter. Copied
//! depth and the reactive mask are actual raster products, so current and previous depth/reactive
//! comparisons use each frame's actual raster coordinates. The model rejection lane is not a
//! rasterized jittered product: it is a conservative mask of padded CPU-projected bounds. The
//! output motion is corrected back to Bevy's unjittered UV convention for reconstruction.
//! [`super::native`] owns the automatic jitter schedule; temporal upscaling remains separate work.

use bevy::camera::MainPassResolutionOverride;
use bevy::core_pipeline::core_3d::graph::{Core3d, Node3d};
use bevy::ecs::query::QueryItem;
use bevy::image::ToExtents;
use bevy::math::DMat4;
use bevy::prelude::*;
use bevy::render::camera::TemporalJitter;
use bevy::render::render_graph::{
    NodeRunError, RenderGraphContext, RenderGraphExt, RenderLabel, ViewNode, ViewNodeRunner,
};
use bevy::render::render_resource::*;
use bevy::render::renderer::{RenderContext, RenderDevice, RenderQueue};
use bevy::render::texture::CachedTexture;
use bevy::render::view::ExtractedView;
use bevy::render::{Render, RenderApp, RenderStartup, RenderSystems};

use super::depth::{self, DepthKey, DepthPublication};
use super::reactive::TemporalReactiveOutput;
use super::rejection::ModelRejectOutput;

/// The camera matrices needed for camera-only reprojection, in Bevy's right-handed view space.
#[derive(Clone, Copy, Debug)]
struct CameraMatrices {
    clip_from_view: Mat4,
    world_from_view: Mat4,
    /// Bevy's jitter offset divided by the physical extent, in UV units.
    jitter_uv: Vec2,
}

impl CameraMatrices {
    fn from_view(
        view: &ExtractedView,
        jitter: Option<&TemporalJitter>,
        size: UVec2,
    ) -> Option<Self> {
        // A custom clip_from_world is an independent projection path; this slice deliberately
        // rejects it rather than mixing it with the ordinary clip_from_view projection.
        if view.clip_from_world.is_some() || size.x == 0 || size.y == 0 {
            return None;
        }
        let world = view.world_from_view.to_matrix();
        let mut clip = view.clip_from_view;
        let mut jitter_uv = Vec2::ZERO;
        if let Some(jitter) = jitter {
            let offset = jitter.offset;
            if !offset.is_finite() || offset.x.abs() > 0.5 || offset.y.abs() > 0.5 {
                return None;
            }
            // Bevy's perspective jitter perturbs clip z_axis.x/y. Orthographic projections use a
            // different perturbation and are not part of this task's supported set.
            if clip.w_axis.w != 0.0 || clip.z_axis.w != -1.0 {
                return None;
            }
            if view.viewport != UVec4::new(0, 0, size.x, size.y) {
                return None;
            }
            if !finite_mat(clip) {
                return None;
            }
            jitter.jitter_projection(&mut clip, size.as_vec2());
            jitter_uv = offset / size.as_vec2();
            if !jitter_uv.is_finite() {
                return None;
            }
        }
        if !finite_mat(clip)
            || !finite_mat(world)
            || clip.determinant().abs() < 1e-12
            || world.determinant().abs() < 1e-12
        {
            return None;
        }
        Some(Self {
            clip_from_view: clip,
            world_from_view: world,
            jitter_uv,
        })
    }
}

/// One view's last encoded camera and, when consecutive, the baseline it may use.
#[derive(Clone, Copy, Debug)]
struct MotionFrames {
    previous: Option<(u64, CameraMatrices)>,
    current: CameraMatrices,
}

impl MotionFrames {
    fn new(current: CameraMatrices) -> Self {
        Self {
            previous: None,
            current,
        }
    }

    /// The previous matrices only when they are exactly the preceding extraction epoch. A wrap or
    /// a gap is a reset, never a simulation-history lookup.
    fn previous_for_epoch(&self, epoch: u64) -> Option<CameraMatrices> {
        self.previous
            .filter(|(previous_epoch, _)| previous_epoch.checked_add(1) == Some(epoch))
            .map(|(_, matrices)| matrices)
    }

    /// Commands for this epoch only make the current cameras the next baseline when both the depth
    /// and motion passes encoded. A skipped frame clears the baseline.
    fn advance(&mut self, epoch: u64, depth_current: bool, motion_current: bool) {
        if depth_current && motion_current {
            self.previous = Some((epoch, self.current));
        } else {
            self.previous = None;
        }
    }
}

/// CPU mirror of the shader's uniform; std140/WGSL layout through `ShaderType`.
#[derive(ShaderType, Clone, Copy, Debug)]
struct MotionUniform {
    /// `inverse(current.clip_from_view)`.
    current_view_from_clip: Mat4,
    /// `previous.clip_from_view * previous_view_from_current_view`.
    previous_clip_from_current_view: Mat4,
    /// `inverse(previous.clip_from_view)`.
    previous_view_from_clip: Mat4,
    /// xy = physical extent, z = previous-frame-usable, w = reserved zero.
    extent_and_flags: Vec4,
    /// xy = previous jitter UV minus current jitter UV; zw = zero.
    raster_jitter_delta: Vec4,
}

/// The six strict texture inputs the gate consumes: current/previous opaque depth and the
/// current/previous masks from both rejection lanes.
struct HistoryInputs<'a> {
    depth_current: &'a TextureView,
    depth_previous: &'a TextureView,
    reactive_current: &'a TextureView,
    reactive_previous: &'a TextureView,
    model_current: &'a TextureView,
    model_previous: &'a TextureView,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum MotionEncodeMode {
    /// Encode a cleared zero-invalid candidate. No pipeline, bind group or draw is used.
    ClearOnly,
    /// All history, pipeline and uniform inputs are ready; encode the fused gate.
    Gated,
}

fn motion_encode_mode(
    inputs_ready: bool,
    pipeline_ready: bool,
    uniform_ready: bool,
) -> MotionEncodeMode {
    if inputs_ready && pipeline_ready && uniform_ready {
        MotionEncodeMode::Gated
    } else {
        MotionEncodeMode::ClearOnly
    }
}

fn finite_mat(matrix: Mat4) -> bool {
    matrix.to_cols_array().iter().all(|value| value.is_finite())
}

fn finite_dmat(matrix: DMat4) -> bool {
    matrix.to_cols_array().iter().all(|value| value.is_finite())
}

/// Build the frame's relative reprojection in f64, then convert the product to f32. The relative
/// transform keeps a large common world translation out of the shader and out of f32 subtraction.
fn camera_reprojection(current: CameraMatrices, previous: CameraMatrices) -> Option<MotionUniform> {
    if !finite_mat(current.clip_from_view)
        || !finite_mat(current.world_from_view)
        || !finite_mat(previous.clip_from_view)
        || !finite_mat(previous.world_from_view)
    {
        return None;
    }
    let relative =
        previous.world_from_view.as_dmat4().inverse() * current.world_from_view.as_dmat4();
    if !finite_dmat(relative) {
        return None;
    }
    let previous_clip_from_current_view = previous.clip_from_view * relative.as_mat4();
    let current_view_from_clip = current.clip_from_view.inverse();
    let previous_view_from_clip = previous.clip_from_view.inverse();
    let raster_jitter_delta = previous.jitter_uv - current.jitter_uv;
    if !finite_mat(previous_clip_from_current_view)
        || !finite_mat(current_view_from_clip)
        || !finite_mat(previous_view_from_clip)
        || !raster_jitter_delta.is_finite()
        || current_view_from_clip.determinant().abs() < 1e-12
        || previous_view_from_clip.determinant().abs() < 1e-12
    {
        return None;
    }
    Some(MotionUniform {
        current_view_from_clip,
        previous_clip_from_current_view,
        previous_view_from_clip,
        extent_and_flags: Vec4::new(0.0, 0.0, 1.0, 0.0),
        raster_jitter_delta: Vec4::new(raster_jitter_delta.x, raster_jitter_delta.y, 0.0, 0.0),
    })
}

/// A zero-motion uniform for a frame with no usable predecessor. The shader also checks the flag,
/// so the identity matrices are belt-and-braces, not a substitute for the flag.
fn zero_motion(current: CameraMatrices, size: UVec2) -> MotionUniform {
    MotionUniform {
        current_view_from_clip: current.clip_from_view.inverse(),
        previous_clip_from_current_view: Mat4::IDENTITY,
        previous_view_from_clip: Mat4::IDENTITY,
        extent_and_flags: Vec4::new(size.x as f32, size.y as f32, 0.0, 0.0),
        raster_jitter_delta: Vec4::ZERO,
    }
}

/// The uniform for one frame and its usable previous baseline. A failed relative reprojection
/// clears the baseline instead of carrying matrices from an unrelated frame.
fn frame_motion(
    frames: &MotionFrames,
    epoch: u64,
    size: UVec2,
) -> (MotionUniform, Option<(u64, CameraMatrices)>) {
    let previous = frames.previous_for_epoch(epoch);
    let reprojected = previous.and_then(|previous| camera_reprojection(frames.current, previous));
    let used_previous = reprojected.is_some();
    let mut uniform = reprojected.unwrap_or_else(|| zero_motion(frames.current, size));
    uniform.extent_and_flags = Vec4::new(
        size.x as f32,
        size.y as f32,
        if used_previous { 1.0 } else { 0.0 },
        0.0,
    );
    (
        uniform,
        if used_previous {
            previous.map(|matrices| (epoch, matrices))
        } else {
            None
        },
    )
}

fn motion_descriptor(size: UVec2) -> TextureDescriptor<'static> {
    TextureDescriptor {
        label: Some("temporal motion candidate"),
        size: size.to_extents(),
        mip_level_count: 1,
        sample_count: 1,
        dimension: TextureDimension::D2,
        format: TextureFormat::Rgba16Float,
        usage: TextureUsages::RENDER_ATTACHMENT | TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    }
}

fn history_inputs<'a>(
    request: &depth::TemporalDepthRequest,
    depth: &'a depth::TemporalDepthTexture,
    reactive: Option<&'a TemporalReactiveOutput>,
    model: Option<&'a ModelRejectOutput>,
    epoch: u64,
) -> Option<HistoryInputs<'a>> {
    let key = request.key();
    if depth.key() != key {
        return None;
    }
    let reactive = reactive.filter(|inputs| inputs.key() == key)?;
    let model = model.filter(|inputs| inputs.key() == key)?;
    Some(HistoryInputs {
        depth_current: depth.current_depth(epoch, request.generation)?,
        depth_previous: depth.previous_depth(epoch, request.generation)?,
        reactive_current: reactive.current_mask(epoch, request.generation)?,
        reactive_previous: reactive.previous_mask(epoch, request.generation)?,
        model_current: model.current_mask(epoch, request.generation)?,
        model_previous: model.previous_mask(epoch, request.generation)?,
    })
}

/// One view's candidate texture, baseline and reusable uniform buffer.
#[derive(Component)]
pub(super) struct TemporalMotionOutput {
    texture: CachedTexture,
    key: DepthKey,
    publication: DepthPublication,
    frames: MotionFrames,
    uniform: UniformBuffer<MotionUniform>,
}

impl TemporalMotionOutput {
    fn new(
        texture: CachedTexture,
        key: DepthKey,
        frames: MotionFrames,
        uniform: UniformBuffer<MotionUniform>,
    ) -> Self {
        Self {
            texture,
            key,
            publication: DepthPublication::new(key.generation),
            frames,
            uniform,
        }
    }

    pub(super) fn current_motion(&self, epoch: u64, generation: u64) -> Option<&TextureView> {
        self.publication
            .is_current(epoch, generation)
            .then_some(&self.texture.default_view)
    }

    pub(super) fn key(&self) -> DepthKey {
        self.key
    }

    pub(super) fn reprojection_binding(&self) -> Option<BindingResource<'_>> {
        self.uniform.binding()
    }

    pub(super) fn has_previous_camera(&self) -> bool {
        self.uniform.get().extent_and_flags.z >= 0.5
    }
}

/// Prepare the candidate allocation and camera baseline after depth preparation. The depth
/// publication deliberately is not required here: the producer has not encoded yet.
#[allow(clippy::type_complexity)] // Bevy query tuples read better inline.
fn prepare_motion_resources(
    mut commands: Commands,
    device: Res<RenderDevice>,
    queue: Res<RenderQueue>,
    requests: Res<depth::TemporalDepthRequests>,
    views: Query<(
        Entity,
        &ExtractedView,
        Option<&TemporalJitter>,
        Option<&MainPassResolutionOverride>,
        &depth::TemporalDepthTexture,
    )>,
    mut motions: Query<(Entity, &mut TemporalMotionOutput)>,
) {
    let epoch = requests.epoch();
    for (render, request) in requests.iter() {
        let Ok((entity, view, jitter, resolution_override, depth_texture)) = views.get(render)
        else {
            commands.entity(render).remove::<TemporalMotionOutput>();
            continue;
        };
        let Some(current) = CameraMatrices::from_view(view, jitter, request.size) else {
            commands.entity(entity).remove::<TemporalMotionOutput>();
            continue;
        };
        if resolution_override.is_some() || depth_texture.key() != request.key() {
            commands.entity(entity).remove::<TemporalMotionOutput>();
            continue;
        }

        if let Ok((_, mut output)) = motions.get_mut(render) {
            if output.key == request.key() {
                let mut frames = output.frames;
                frames.current = current;
                let (uniform, previous) = frame_motion(&frames, epoch, request.size);
                frames.previous = previous;
                output.frames = frames;
                output.uniform.set(uniform);
                output.uniform.write_buffer(&device, &queue);
                continue;
            }
        }

        let frames = MotionFrames::new(current);
        let (uniform, previous) = frame_motion(&frames, epoch, request.size);
        let frames = MotionFrames { previous, current };
        let texture = device.create_texture(&motion_descriptor(request.size));
        let default_view = texture.create_view(&TextureViewDescriptor::default());
        let mut uniform_buffer = UniformBuffer::from(uniform);
        uniform_buffer.write_buffer(&device, &queue);
        commands.entity(entity).insert(TemporalMotionOutput::new(
            CachedTexture {
                texture,
                default_view,
            },
            request.key(),
            frames,
            uniform_buffer,
        ));
    }

    for (entity, _) in &mut motions {
        if !requests.contains(entity) {
            commands.entity(entity).remove::<TemporalMotionOutput>();
        }
    }
}

#[derive(Resource)]
struct TemporalMotionPipeline {
    layout: BindGroupLayoutDescriptor,
    pipeline: CachedRenderPipelineId,
}

fn init_motion_pipeline(
    mut commands: Commands,
    asset_server: Option<Res<AssetServer>>,
    cache: Res<PipelineCache>,
) {
    let Some(asset_server) = asset_server else {
        return;
    };
    let texture_entry = |binding| BindGroupLayoutEntry {
        binding,
        visibility: ShaderStages::FRAGMENT,
        ty: BindingType::Texture {
            sample_type: TextureSampleType::Float { filterable: false },
            view_dimension: TextureViewDimension::D2,
            multisampled: false,
        },
        count: None,
    };
    let layout = BindGroupLayoutDescriptor::new(
        "temporal motion",
        &[
            texture_entry(0),
            BindGroupLayoutEntry {
                binding: 1,
                visibility: ShaderStages::FRAGMENT,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            texture_entry(2),
            texture_entry(3),
            texture_entry(4),
            texture_entry(5),
            texture_entry(6),
        ],
    );
    let shader: Handle<Shader> =
        asset_server.load("embedded://benilla_world/shaders/temporal_motion.wgsl");
    let pipeline = cache.queue_render_pipeline(RenderPipelineDescriptor {
        label: Some("temporal motion candidate".into()),
        layout: vec![layout.clone()],
        vertex: VertexState {
            shader: shader.clone(),
            entry_point: Some("vertex".into()),
            ..default()
        },
        fragment: Some(FragmentState {
            shader,
            entry_point: Some("fragment".into()),
            targets: vec![Some(ColorTargetState {
                format: TextureFormat::Rgba16Float,
                blend: None,
                write_mask: ColorWrites::ALL,
            })],
            ..default()
        }),
        primitive: PrimitiveState {
            topology: PrimitiveTopology::TriangleList,
            ..default()
        },
        multisample: MultisampleState {
            count: 1,
            ..default()
        },
        ..default()
    });
    commands.insert_resource(TemporalMotionPipeline { layout, pipeline });
}

#[derive(Debug, Hash, PartialEq, Eq, Clone, RenderLabel)]
pub(super) struct TemporalMotionLabel;

#[derive(Default)]
struct TemporalMotionNode;

impl ViewNode for TemporalMotionNode {
    type ViewQuery = (
        &'static depth::TemporalDepthTexture,
        &'static TemporalMotionOutput,
        Option<&'static TemporalReactiveOutput>,
        Option<&'static ModelRejectOutput>,
    );

    fn run<'w>(
        &self,
        graph: &mut RenderGraphContext,
        context: &mut RenderContext<'w>,
        (depth, motion, reactive, model): QueryItem<'w, '_, Self::ViewQuery>,
        world: &'w World,
    ) -> Result<(), NodeRunError> {
        let requests = world.resource::<depth::TemporalDepthRequests>();
        let Some(request) = requests.get(graph.view_entity()) else {
            return Ok(());
        };
        if depth.key() != request.key() || motion.key != request.key() {
            return Ok(());
        }
        let epoch = requests.epoch();
        if depth.current_depth(epoch, request.generation).is_none() {
            return Ok(());
        }

        let history = history_inputs(request, depth, reactive, model, epoch);
        let pipeline = world.get_resource::<TemporalMotionPipeline>();
        let cache = world.resource::<PipelineCache>();
        let render_pipeline =
            pipeline.and_then(|pipeline| cache.get_render_pipeline(pipeline.pipeline));
        let uniform_binding = motion.uniform.binding();
        let mode = motion_encode_mode(
            history.is_some(),
            render_pipeline.is_some(),
            uniform_binding.is_some(),
        );
        let device = world.resource::<RenderDevice>();
        let destination = &motion.texture.default_view;
        let encoded = motion.publication.encode(epoch, request.generation, || {
            let mut pass = context
                .command_encoder()
                .begin_render_pass(&RenderPassDescriptor {
                    label: Some("temporal camera-motion candidate"),
                    color_attachments: &[Some(RenderPassColorAttachment {
                        view: destination,
                        resolve_target: None,
                        depth_slice: None,
                        ops: Operations {
                            load: LoadOp::Clear(Default::default()),
                            store: StoreOp::Store,
                        },
                    })],
                    ..default()
                });
            if mode == MotionEncodeMode::Gated {
                if let (
                    Some(history),
                    Some(pipeline),
                    Some(render_pipeline),
                    Some(uniform_binding),
                ) = (history, pipeline, render_pipeline, uniform_binding)
                {
                    let bind = device.create_bind_group(
                        "temporal motion",
                        &cache.get_bind_group_layout(&pipeline.layout),
                        &BindGroupEntries::sequential((
                            history.depth_current,
                            uniform_binding,
                            history.depth_previous,
                            history.reactive_current,
                            history.reactive_previous,
                            history.model_current,
                            history.model_previous,
                        )),
                    );
                    pass.set_pipeline(render_pipeline);
                    pass.set_bind_group(0, &bind, &[]);
                    pass.draw(0..3, 0..1);
                }
            }
            drop(pass);
            true
        });
        if encoded {
            debug_assert!(motion.current_motion(epoch, request.generation).is_some());
        }
        Ok(())
    }
}

/// Advance the baseline only after both producers recorded their commands. A skipped frame clears
/// it; preparation also checks consecutive epochs before using it.
#[allow(clippy::type_complexity)] // Bevy query tuples read better inline.
fn finish_motion_frame(
    requests: Res<depth::TemporalDepthRequests>,
    depths: Query<&depth::TemporalDepthTexture>,
    mut motions: Query<(Entity, &mut TemporalMotionOutput)>,
) {
    let epoch = requests.epoch();
    for (entity, mut motion) in &mut motions {
        let Some(request) = requests.get(entity) else {
            motion.frames.previous = None;
            continue;
        };
        let depth_current = depths.get(entity).is_ok_and(|depth| {
            depth.key() == request.key() && depth.current_depth(epoch, request.generation).is_some()
        });
        let motion_current =
            motion.key == request.key() && motion.publication.is_current(epoch, request.generation);
        motion.frames.advance(epoch, depth_current, motion_current);
    }
}

/// Private child plugin: registered from [`super::AaPlugin`].
pub(super) struct TemporalMotionPlugin;

impl Plugin for TemporalMotionPlugin {
    fn build(&self, app: &mut App) {
        let Some(render_app) = app.get_sub_app_mut(RenderApp) else {
            return;
        };
        render_app
            .add_systems(
                Render,
                prepare_motion_resources
                    .after(depth::prepare_depth_textures)
                    .in_set(RenderSystems::PrepareBindGroups),
            )
            .add_systems(Render, finish_motion_frame.in_set(RenderSystems::Cleanup))
            .add_systems(RenderStartup, init_motion_pipeline)
            .add_render_graph_node::<ViewNodeRunner<TemporalMotionNode>>(
                Core3d,
                TemporalMotionLabel,
            )
            .add_render_graph_edges(
                Core3d,
                (
                    depth::TemporalDepthLabel,
                    TemporalMotionLabel,
                    Node3d::EndMainPass,
                ),
            )
            .add_render_graph_edges(
                Core3d,
                (super::reactive::TemporalReactiveLabel, TemporalMotionLabel),
            );
    }
}

#[cfg(test)]
pub(super) fn gpu_probe(
    world: &World,
    render: Entity,
    epoch: u64,
    generation: u64,
) -> Option<super::gpu_validation::TextureProbe> {
    let output = world.get::<TemporalMotionOutput>(render)?;
    Some(super::gpu_validation::TextureProbe {
        key: output.key,
        current: output
            .publication
            .is_current(epoch, generation)
            .then(|| output.texture.texture.clone()),
        previous: None,
    })
}

#[cfg(test)]
pub(super) fn gpu_frame_diagnostics(world: &World, render: Entity) -> String {
    // Post-frame state only. Do not infer the node's prior Gated/ClearOnly decision from
    // camera-baseline fields that Cleanup has already advanced.
    let Some(requests) = world.get_resource::<depth::TemporalDepthRequests>() else {
        return "motion frame: requests absent".into();
    };
    let Some(request) = requests.get(render) else {
        return format!("motion frame: no request for {render:?}");
    };
    let epoch = requests.epoch();
    let key = request.key();
    let output = world.get::<TemporalMotionOutput>(render);
    let depth = world.get::<depth::TemporalDepthTexture>(render);
    let reactive = world.get::<TemporalReactiveOutput>(render);
    let model = world.get::<ModelRejectOutput>(render);
    let inputs_ready = depth
        .and_then(|depth| history_inputs(request, depth, reactive, model, epoch))
        .is_some();
    let publication = output
        .map(|output| output.publication.is_current(epoch, request.generation))
        .unwrap_or(false);
    let output_key = output.map(|output| output.key);
    let uniform = output.map(|output| {
        (
            output.uniform.get().extent_and_flags.z,
            output.uniform.binding().is_some(),
        )
    });
    let pipeline = world
        .get_resource::<TemporalMotionPipeline>()
        .map(|pipeline| {
            world
                .resource::<PipelineCache>()
                .get_render_pipeline_state(pipeline.pipeline)
        });
    format!(
        "request={key:?} epoch={epoch} generation={} output_key={output_key:?} publication={publication} history_inputs={inputs_ready} uniform_usable/binding={uniform:?} pipeline={pipeline:?}",
        request.generation,
    )
}

#[cfg(test)]
pub(super) fn gpu_pipeline_ready(world: &World) -> bool {
    world
        .get_resource::<TemporalMotionPipeline>()
        .and_then(|pipeline| {
            world
                .get_resource::<PipelineCache>()?
                .get_render_pipeline(pipeline.pipeline)
        })
        .is_some()
}

#[cfg(test)]
pub(super) fn gpu_pipeline_diagnostics(world: &World) -> Vec<String> {
    let Some(pipeline) = world.get_resource::<TemporalMotionPipeline>() else {
        return vec!["motion pipeline: resource absent".into()];
    };
    let cache = world.resource::<PipelineCache>();
    vec![format!(
        "motion pipeline: {:?}",
        cache.get_render_pipeline_state(pipeline.pipeline)
    )]
}

#[cfg(test)]
mod tests;
