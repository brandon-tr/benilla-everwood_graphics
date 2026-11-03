//! Bevy-derived native temporal resolve with independently owned reconstructed history.
//!
//! The pass runs only for a camera carrying [`TemporalResolveView`]; `AaPlugin` maintains that
//! marker from the effective AA mode, so production cameras opt in through `gxAntiAliasing`.
//!
//! The two RGBA16Float history slots add 16 bytes/pixel per opt-in view, excluding allocator
//! overhead. This pass intentionally differs from stock Bevy TAA: manual bilinear history filtering
//! and explicit filter-footprint rejection replace the higher-order history filter, and the
//! reconstructed history is separate from the raw scene-color captures. No performance savings are
//! claimed.

use bevy::core_pipeline::core_3d::graph::{Core3d, Node3d};
use bevy::ecs::query::QueryItem;
use bevy::image::ToExtents;
use bevy::prelude::*;
use bevy::render::render_graph::{
    NodeRunError, RenderGraphContext, RenderGraphExt, RenderLabel, ViewNode, ViewNodeRunner,
};
use bevy::render::render_resource::*;
use bevy::render::renderer::{RenderContext, RenderDevice};
use bevy::render::sync_world::RenderEntity;
use bevy::render::texture::CachedTexture;
use bevy::render::view::ViewTarget;
use bevy::render::{Extract, ExtractSchedule, Render, RenderApp, RenderStartup, RenderSystems};

use super::depth::{self, DepthKey};
use super::frame_pair::TemporalFramePair;
use super::motion::TemporalMotionOutput;
use super::native::NativeResolveBridge;
use super::reactive::{self, TemporalReactiveOutput};
use super::rejection::ModelRejectOutput;
use super::scene_color::TemporalSceneColorLabel;

/// The production opt-in marker for the native resolve.
///
/// [`super::AaPlugin`] inserts it on a [`crate::view::WorldCamera`] exactly while the effective AA
/// mode is [`super::AaEffective::Taa`] and removes it otherwise, so no camera has to opt in by hand.
/// A camera without the marker allocates no history and encodes no resolve pass.
#[derive(Component, Clone, Default)]
pub struct TemporalResolveView;

#[derive(Component)]
pub(super) struct TemporalResolveHistory {
    key: DepthKey,
    frames: TemporalFramePair<CachedTexture>,
}

impl TemporalResolveHistory {
    fn new(key: DepthKey, textures: [CachedTexture; 2]) -> Self {
        Self {
            key,
            frames: TemporalFramePair::new(key.generation, textures),
        }
    }

    pub(super) fn key(&self) -> DepthKey {
        self.key
    }

    #[allow(dead_code)] // Future resolve consumers; GPU test probes use these directly.
    pub(super) fn current_history(&self, epoch: u64, generation: u64) -> Option<&TextureView> {
        self.frames
            .current(epoch, generation)
            .map(|texture| &texture.default_view)
    }

    #[allow(dead_code)] // Future resolve consumers; GPU test probes use these directly.
    pub(super) fn previous_history(&self, epoch: u64, generation: u64) -> Option<&TextureView> {
        self.frames
            .previous(epoch, generation)
            .map(|texture| &texture.default_view)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum ResolveDecision {
    Keep,
    Replace,
    Remove,
}

pub(super) fn resolve_decision(
    marker: bool,
    existing: Option<DepthKey>,
    requested: DepthKey,
    source_valid: bool,
) -> ResolveDecision {
    if !marker || !source_valid {
        ResolveDecision::Remove
    } else if existing == Some(requested) {
        ResolveDecision::Keep
    } else {
        ResolveDecision::Replace
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum ResolveMode {
    Seed,
    Temporal,
    Skip,
}

#[allow(clippy::too_many_arguments)]
pub(super) fn resolve_mode(
    marker: bool,
    current_inputs: bool,
    prior_history: bool,
    temporal_inputs: bool,
    seed_pipeline: bool,
    temporal_pipeline: bool,
) -> ResolveMode {
    if !marker || !current_inputs {
        return ResolveMode::Skip;
    }
    if prior_history && temporal_inputs {
        return if temporal_pipeline {
            ResolveMode::Temporal
        } else {
            ResolveMode::Skip
        };
    }
    if seed_pipeline {
        ResolveMode::Seed
    } else {
        ResolveMode::Skip
    }
}

fn history_descriptor(size: UVec2) -> TextureDescriptor<'static> {
    TextureDescriptor {
        label: Some("temporal reconstructed history"),
        size: size.to_extents(),
        mip_level_count: 1,
        sample_count: 1,
        dimension: TextureDimension::D2,
        format: TextureFormat::Rgba16Float,
        usage: TextureUsages::RENDER_ATTACHMENT | TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    }
}

#[allow(clippy::type_complexity)]
fn prepare_resolve_resources(
    mut commands: Commands,
    device: Res<RenderDevice>,
    requests: Res<depth::TemporalDepthRequests>,
    views: Query<(
        Entity,
        Option<&TemporalResolveView>,
        &ViewTarget,
        &depth::TemporalDepthTexture,
        Option<&TemporalResolveHistory>,
    )>,
    outputs: Query<(Entity, &TemporalResolveHistory)>,
) {
    for (render, request) in requests.iter() {
        let Ok((entity, marker, view, depth_texture, existing)) = views.get(render) else {
            commands.entity(render).remove::<TemporalResolveHistory>();
            continue;
        };
        let source_valid =
            depth_texture.key() == request.key() && reactive::source_matches(view, request);
        match resolve_decision(
            marker.is_some(),
            existing.map(TemporalResolveHistory::key),
            request.key(),
            source_valid,
        ) {
            ResolveDecision::Remove => {
                commands.entity(entity).remove::<TemporalResolveHistory>();
            }
            ResolveDecision::Keep => {}
            ResolveDecision::Replace => {
                let a = device.create_texture(&history_descriptor(request.size));
                let b = device.create_texture(&history_descriptor(request.size));
                commands.entity(entity).insert(TemporalResolveHistory::new(
                    request.key(),
                    [
                        CachedTexture {
                            default_view: a.create_view(&TextureViewDescriptor::default()),
                            texture: a,
                        },
                        CachedTexture {
                            default_view: b.create_view(&TextureViewDescriptor::default()),
                            texture: b,
                        },
                    ],
                ));
            }
        }
    }
    for (entity, _) in &outputs {
        if !requests.contains(entity) {
            commands.entity(entity).remove::<TemporalResolveHistory>();
        }
    }
}

#[derive(Resource)]
struct ResolvePipelines {
    seed_layout: BindGroupLayoutDescriptor,
    seed: CachedRenderPipelineId,
    temporal_layout: BindGroupLayoutDescriptor,
    temporal: CachedRenderPipelineId,
}

fn texture_entry(binding: u32) -> BindGroupLayoutEntry {
    BindGroupLayoutEntry {
        binding,
        visibility: ShaderStages::FRAGMENT,
        ty: BindingType::Texture {
            sample_type: TextureSampleType::Float { filterable: false },
            view_dimension: TextureViewDimension::D2,
            multisampled: false,
        },
        count: None,
    }
}

fn temporal_layout_entries() -> [BindGroupLayoutEntry; 10] {
    [
        texture_entry(0),
        texture_entry(1),
        texture_entry(2),
        texture_entry(3),
        texture_entry(4),
        texture_entry(5),
        texture_entry(6),
        texture_entry(7),
        texture_entry(8),
        BindGroupLayoutEntry {
            binding: 9,
            visibility: ShaderStages::FRAGMENT,
            ty: BindingType::Buffer {
                ty: BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        },
    ]
}

fn resolve_targets() -> Vec<Option<ColorTargetState>> {
    vec![
        Some(ColorTargetState {
            format: TextureFormat::Rgba16Float,
            blend: None,
            write_mask: ColorWrites::ALL,
        }),
        Some(ColorTargetState {
            format: TextureFormat::Rgba16Float,
            blend: None,
            write_mask: ColorWrites::ALL,
        }),
    ]
}

fn init_resolve_pipelines(
    mut commands: Commands,
    asset_server: Option<Res<AssetServer>>,
    cache: Res<PipelineCache>,
) {
    let Some(asset_server) = asset_server else {
        return;
    };
    let shader: Handle<Shader> =
        asset_server.load("embedded://benilla_world/shaders/temporal_resolve.wgsl");
    let seed_layout = BindGroupLayoutDescriptor::new("temporal resolve seed", &[texture_entry(0)]);
    let seed = cache.queue_render_pipeline(RenderPipelineDescriptor {
        label: Some("temporal resolve seed".into()),
        layout: vec![seed_layout.clone()],
        vertex: VertexState {
            shader: shader.clone(),
            entry_point: Some("vertex".into()),
            ..default()
        },
        fragment: Some(FragmentState {
            shader: shader.clone(),
            entry_point: Some("seed_fragment".into()),
            targets: resolve_targets(),
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
    let temporal_layout =
        BindGroupLayoutDescriptor::new("temporal resolve", &temporal_layout_entries());
    let temporal = cache.queue_render_pipeline(RenderPipelineDescriptor {
        label: Some("temporal resolve".into()),
        layout: vec![temporal_layout.clone()],
        vertex: VertexState {
            shader: shader.clone(),
            entry_point: Some("vertex".into()),
            ..default()
        },
        fragment: Some(FragmentState {
            shader,
            entry_point: Some("temporal_fragment".into()),
            targets: resolve_targets(),
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
    commands.insert_resource(ResolvePipelines {
        seed_layout,
        seed,
        temporal_layout,
        temporal,
    });
}

#[derive(Debug, Hash, PartialEq, Eq, Clone, RenderLabel)]
pub(super) struct TemporalResolveLabel;

#[derive(Default)]
struct TemporalResolveNode;

impl ViewNode for TemporalResolveNode {
    type ViewQuery = (
        Option<&'static TemporalResolveView>,
        &'static ViewTarget,
        &'static TemporalResolveHistory,
        &'static depth::TemporalDepthTexture,
        Option<&'static TemporalMotionOutput>,
        Option<&'static TemporalReactiveOutput>,
        Option<&'static ModelRejectOutput>,
    );

    fn run<'w>(
        &self,
        graph: &mut RenderGraphContext,
        context: &mut RenderContext<'w>,
        (marker, view_target, history, depth, motion, reactive, model): QueryItem<
            'w,
            '_,
            Self::ViewQuery,
        >,
        world: &'w World,
    ) -> Result<(), NodeRunError> {
        if marker.is_none() {
            return Ok(());
        }
        let bridge = world.resource::<NativeResolveBridge>();
        let requests = world.resource::<depth::TemporalDepthRequests>();
        let Some(request) = requests.get(graph.view_entity()) else {
            return Ok(());
        };
        let epoch = requests.epoch();
        if history.key() != request.key() || depth.key() != request.key() {
            return Ok(());
        }
        if !reactive::source_matches(view_target, request) {
            return Ok(());
        }
        let Some(current_depth) = depth.current_depth(epoch, request.generation) else {
            return Ok(());
        };
        let Some(motion) = motion else {
            return Ok(());
        };
        if motion.key() != request.key() {
            return Ok(());
        }
        let Some(motion_texture) = motion.current_motion(epoch, request.generation) else {
            return Ok(());
        };
        let (Some(reactive), Some(model)) = (reactive, model) else {
            return Ok(());
        };
        if reactive.key() != request.key() || model.key() != request.key() {
            return Ok(());
        }
        let Some(current_reactive) = reactive.current_mask(epoch, request.generation) else {
            return Ok(());
        };
        let Some(current_model) = model.current_mask(epoch, request.generation) else {
            return Ok(());
        };
        let Some(pipelines) = world.get_resource::<ResolvePipelines>() else {
            return Ok(());
        };
        let cache = world.resource::<PipelineCache>();
        let seed = cache.get_render_pipeline(pipelines.seed);
        let temporal_pipeline = cache.get_render_pipeline(pipelines.temporal);
        let seed_layout = cache.get_bind_group_layout(&pipelines.seed_layout);
        let temporal_layout = cache.get_bind_group_layout(&pipelines.temporal_layout);
        let uniform = motion.reprojection_binding();
        let previous_history = history
            .frames
            .previous_for_write(epoch, request.generation)
            .map(|texture| &texture.default_view);
        let previous_depth = depth.previous_depth(epoch, request.generation);
        let previous_reactive = reactive.previous_mask(epoch, request.generation);
        let previous_model = model.previous_mask(epoch, request.generation);
        let temporal_inputs = previous_depth.is_some()
            && previous_reactive.is_some()
            && previous_model.is_some()
            && motion.has_previous_camera()
            && uniform.is_some();
        let mode = resolve_mode(
            marker.is_some(),
            true,
            previous_history.is_some(),
            temporal_inputs,
            seed.is_some(),
            temporal_pipeline.is_some(),
        );
        let device = world.resource::<RenderDevice>();

        match mode {
            ResolveMode::Skip => Ok(()),
            ResolveMode::Seed => {
                let Some(seed) = seed else {
                    return Ok(());
                };
                let encoded =
                    history
                        .frames
                        .encode_current(epoch, request.generation, |destination| {
                            let post = view_target.post_process_write();
                            let bind = device.create_bind_group(
                                "temporal resolve seed",
                                &seed_layout,
                                &BindGroupEntries::single(post.source),
                            );
                            let mut pass = context.command_encoder().begin_render_pass(
                                &RenderPassDescriptor {
                                    label: Some("temporal resolve seed"),
                                    color_attachments: &[
                                        Some(RenderPassColorAttachment {
                                            view: post.destination,
                                            resolve_target: None,
                                            depth_slice: None,
                                            ops: Operations {
                                                load: LoadOp::Clear(Default::default()),
                                                store: StoreOp::Store,
                                            },
                                        }),
                                        Some(RenderPassColorAttachment {
                                            view: &destination.default_view,
                                            resolve_target: None,
                                            depth_slice: None,
                                            ops: Operations {
                                                load: LoadOp::Clear(Default::default()),
                                                store: StoreOp::Store,
                                            },
                                        }),
                                    ],
                                    ..default()
                                },
                            );
                            pass.set_pipeline(seed);
                            pass.set_bind_group(0, &bind, &[]);
                            pass.draw(0..3, 0..1);
                            true
                        });
                if encoded {
                    let key = history.key();
                    bridge.mark_committed(key.main_view, epoch, key.generation);
                }
                Ok(())
            }
            ResolveMode::Temporal => {
                let (
                    Some(pipeline),
                    Some(previous_history),
                    Some(previous_depth),
                    Some(previous_reactive),
                    Some(previous_model),
                    Some(uniform),
                ) = (
                    temporal_pipeline,
                    previous_history,
                    previous_depth,
                    previous_reactive,
                    previous_model,
                    uniform,
                )
                else {
                    return Ok(());
                };
                let encoded =
                    history
                        .frames
                        .encode_current(epoch, request.generation, |destination| {
                            let post = view_target.post_process_write();
                            let bind = device.create_bind_group(
                                "temporal resolve",
                                &temporal_layout,
                                &BindGroupEntries::sequential((
                                    post.source,
                                    previous_history,
                                    motion_texture,
                                    current_depth,
                                    previous_depth,
                                    current_reactive,
                                    previous_reactive,
                                    current_model,
                                    previous_model,
                                    uniform,
                                )),
                            );
                            let mut pass = context.command_encoder().begin_render_pass(
                                &RenderPassDescriptor {
                                    label: Some("temporal resolve"),
                                    color_attachments: &[
                                        Some(RenderPassColorAttachment {
                                            view: post.destination,
                                            resolve_target: None,
                                            depth_slice: None,
                                            ops: Operations {
                                                load: LoadOp::Clear(Default::default()),
                                                store: StoreOp::Store,
                                            },
                                        }),
                                        Some(RenderPassColorAttachment {
                                            view: &destination.default_view,
                                            resolve_target: None,
                                            depth_slice: None,
                                            ops: Operations {
                                                load: LoadOp::Clear(Default::default()),
                                                store: StoreOp::Store,
                                            },
                                        }),
                                    ],
                                    ..default()
                                },
                            );
                            pass.set_pipeline(pipeline);
                            pass.set_bind_group(0, &bind, &[]);
                            pass.draw(0..3, 0..1);
                            true
                        });
                if encoded {
                    let key = history.key();
                    bridge.mark_committed(key.main_view, epoch, key.generation);
                }
                Ok(())
            }
        }
    }
}

#[cfg(test)]
pub(super) struct ResolveProbe {
    pub(super) key: DepthKey,
    pub(super) current: Option<Texture>,
    pub(super) previous: Option<Texture>,
}

#[cfg(test)]
pub(super) fn gpu_probe(
    world: &World,
    render: Entity,
    epoch: u64,
    generation: u64,
) -> Option<ResolveProbe> {
    let history = world.get::<TemporalResolveHistory>(render)?;
    Some(ResolveProbe {
        key: history.key(),
        current: history
            .frames
            .current(epoch, generation)
            .map(|texture| texture.texture.clone()),
        previous: history
            .frames
            .previous(epoch, generation)
            .map(|texture| texture.texture.clone()),
    })
}

#[cfg(test)]
pub(super) fn gpu_pipeline_ready(world: &World) -> bool {
    let Some(pipelines) = world.get_resource::<ResolvePipelines>() else {
        return false;
    };
    let Some(cache) = world.get_resource::<PipelineCache>() else {
        return false;
    };
    cache.get_render_pipeline(pipelines.seed).is_some()
        && cache.get_render_pipeline(pipelines.temporal).is_some()
}

#[cfg(test)]
#[allow(dead_code)] // Test-only timeout diagnostics.
pub(super) fn gpu_pipeline_diagnostics(world: &World) -> Vec<String> {
    let Some(pipelines) = world.get_resource::<ResolvePipelines>() else {
        return vec!["resolve pipelines: resource absent".into()];
    };
    let cache = world.resource::<PipelineCache>();
    vec![
        format!(
            "resolve seed: {:?}",
            cache.get_render_pipeline_state(pipelines.seed)
        ),
        format!(
            "resolve temporal: {:?}",
            cache.get_render_pipeline_state(pipelines.temporal)
        ),
    ]
}

/// Copy the production marker onto the render view. This is a plain render-world component, not a
/// `SyncToRenderWorld` component: a zero-sized marker needs none of the sync machinery, and a
/// camera in an app without a render world (a unit-test harness) can carry or drop it freely.
#[allow(clippy::type_complexity)] // Bevy query tuples read better inline.
fn extract_native_marker(
    mut commands: Commands,
    cameras: Extract<
        Query<(Entity, &RenderEntity, Has<TemporalResolveView>), With<bevy::camera::Camera3d>>,
    >,
) {
    for (_main, render, marked) in cameras.iter() {
        if marked {
            commands.entity(render.id()).insert(TemporalResolveView);
        } else {
            commands.entity(render.id()).remove::<TemporalResolveView>();
        }
    }
}

pub(super) struct TemporalResolvePlugin;

impl Plugin for TemporalResolvePlugin {
    fn build(&self, app: &mut App) {
        let Some(render_app) = app.get_sub_app_mut(RenderApp) else {
            return;
        };
        render_app
            .add_systems(ExtractSchedule, extract_native_marker)
            .add_systems(
                Render,
                prepare_resolve_resources
                    .after(depth::prepare_depth_textures)
                    .in_set(RenderSystems::PrepareBindGroups),
            )
            .add_systems(RenderStartup, init_resolve_pipelines)
            .add_render_graph_node::<ViewNodeRunner<TemporalResolveNode>>(
                Core3d,
                TemporalResolveLabel,
            )
            .add_render_graph_edges(
                Core3d,
                (
                    TemporalSceneColorLabel,
                    TemporalResolveLabel,
                    Node3d::EndMainPass,
                ),
            );
    }
}

#[cfg(test)]
mod tests;
