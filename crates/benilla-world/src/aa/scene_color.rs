//! Persistent current/previous assembled scene-color captures at the native reconstruction boundary.
//!
//! The capture is recorded after motion and before `EndMainPass`, so it contains the assembled
//! opaque/transmissive/transparent scene before app-owned volumetric fog/light compositing. It is a
//! capture foundation, not reconstructed output or committed temporal history; nothing samples or
//! blends previous color here.

use bevy::core_pipeline::core_3d::graph::{Core3d, Node3d};
use bevy::ecs::query::QueryItem;
use bevy::prelude::*;
use bevy::render::render_graph::{
    NodeRunError, RenderGraphContext, RenderGraphExt, RenderLabel, ViewNode, ViewNodeRunner,
};
use bevy::render::render_resource::*;
use bevy::render::renderer::{RenderContext, RenderDevice};
use bevy::render::texture::CachedTexture;
use bevy::render::view::ViewTarget;
use bevy::render::{Render, RenderApp, RenderStartup, RenderSystems};

use super::depth::{self, DepthKey};
use super::frame_pair::TemporalFramePair;
use super::motion::TemporalMotionLabel;
use super::reactive::{reactive_descriptor, source_matches};

/// One view's two independently owned assembled-color captures.
#[derive(Component)]
pub(super) struct TemporalSceneColor {
    key: DepthKey,
    frames: TemporalFramePair<CachedTexture>,
}

impl TemporalSceneColor {
    fn new(key: DepthKey, textures: [CachedTexture; 2]) -> Self {
        Self {
            key,
            frames: TemporalFramePair::new(key.generation, textures),
        }
    }

    pub(super) fn key(&self) -> DepthKey {
        self.key
    }

    pub(super) fn current_color(&self, epoch: u64, generation: u64) -> Option<&TextureView> {
        self.frames
            .current(epoch, generation)
            .map(|color| &color.default_view)
    }

    /// The immediately preceding capture; the future native resolve consumes it.
    #[allow(dead_code)] // Future resolve accessor; no runtime consumer is fabricated here.
    pub(super) fn previous_color(&self, epoch: u64, generation: u64) -> Option<&TextureView> {
        self.frames
            .previous(epoch, generation)
            .map(|color| &color.default_view)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum SceneColorDecision {
    Keep,
    Replace,
    Remove,
}

pub(super) fn scene_color_decision(
    existing: Option<DepthKey>,
    requested: DepthKey,
    source_valid: bool,
) -> SceneColorDecision {
    if !source_valid {
        SceneColorDecision::Remove
    } else if existing == Some(requested) {
        SceneColorDecision::Keep
    } else {
        SceneColorDecision::Replace
    }
}

pub(super) fn scene_color_descriptor(size: UVec2) -> TextureDescriptor<'static> {
    reactive_descriptor(
        size,
        TextureFormat::Rgba16Float,
        "temporal assembled scene color",
    )
}

#[allow(clippy::type_complexity)] // Bevy query tuples read better inline.
fn prepare_scene_color_resources(
    mut commands: Commands,
    device: Res<RenderDevice>,
    requests: Res<depth::TemporalDepthRequests>,
    views: Query<(
        Entity,
        &ViewTarget,
        &depth::TemporalDepthTexture,
        Option<&TemporalSceneColor>,
    )>,
    outputs: Query<(Entity, &TemporalSceneColor)>,
) {
    for (render, request) in requests.iter() {
        let Ok((entity, view, depth_texture, existing)) = views.get(render) else {
            commands.entity(render).remove::<TemporalSceneColor>();
            continue;
        };
        let source_valid = depth_texture.key() == request.key() && source_matches(view, request);
        match scene_color_decision(
            existing.map(TemporalSceneColor::key),
            request.key(),
            source_valid,
        ) {
            SceneColorDecision::Remove => {
                commands.entity(entity).remove::<TemporalSceneColor>();
            }
            SceneColorDecision::Keep => {}
            SceneColorDecision::Replace => {
                // Two independent persistent captures. This is new foundation cost, not a
                // performance saving: 16 bytes/pixel excluding allocator overhead.
                let color_a = device.create_texture(&scene_color_descriptor(request.size));
                let color_b = device.create_texture(&scene_color_descriptor(request.size));
                let color_a = CachedTexture {
                    default_view: color_a.create_view(&TextureViewDescriptor::default()),
                    texture: color_a,
                };
                let color_b = CachedTexture {
                    default_view: color_b.create_view(&TextureViewDescriptor::default()),
                    texture: color_b,
                };
                commands
                    .entity(entity)
                    .insert(TemporalSceneColor::new(request.key(), [color_a, color_b]));
            }
        }
    }

    for (entity, _) in &outputs {
        if !requests.contains(entity) {
            commands.entity(entity).remove::<TemporalSceneColor>();
        }
    }
}

#[derive(Resource)]
struct SceneColorPipeline {
    layout: BindGroupLayoutDescriptor,
    pipeline: CachedRenderPipelineId,
}

fn init_scene_color_pipeline(
    mut commands: Commands,
    asset_server: Option<Res<AssetServer>>,
    cache: Res<PipelineCache>,
) {
    let Some(asset_server) = asset_server else {
        return;
    };
    let layout = BindGroupLayoutDescriptor::new(
        "temporal assembled scene color",
        &[BindGroupLayoutEntry {
            binding: 0,
            visibility: ShaderStages::FRAGMENT,
            ty: BindingType::Texture {
                sample_type: TextureSampleType::Float { filterable: false },
                view_dimension: TextureViewDimension::D2,
                multisampled: false,
            },
            count: None,
        }],
    );
    let shader: Handle<Shader> =
        asset_server.load("embedded://benilla_world/shaders/temporal_reactive.wgsl");
    let pipeline = cache.queue_render_pipeline(RenderPipelineDescriptor {
        label: Some("temporal assembled scene color".into()),
        layout: vec![layout.clone()],
        vertex: VertexState {
            shader: shader.clone(),
            entry_point: Some("vertex".into()),
            ..default()
        },
        fragment: Some(FragmentState {
            shader,
            entry_point: Some("copy_fragment".into()),
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
    commands.insert_resource(SceneColorPipeline { layout, pipeline });
}

#[derive(Debug, Hash, PartialEq, Eq, Clone, RenderLabel)]
pub(super) struct TemporalSceneColorLabel;

#[derive(Default)]
struct TemporalSceneColorNode;

impl ViewNode for TemporalSceneColorNode {
    type ViewQuery = (&'static ViewTarget, &'static TemporalSceneColor);

    fn run<'w>(
        &self,
        graph: &mut RenderGraphContext,
        context: &mut RenderContext<'w>,
        (view_target, output): QueryItem<'w, '_, Self::ViewQuery>,
        world: &'w World,
    ) -> Result<(), NodeRunError> {
        let requests = world.resource::<depth::TemporalDepthRequests>();
        let Some(request) = requests.get(graph.view_entity()) else {
            return Ok(());
        };
        if output.key() != request.key() || !source_matches(view_target, request) {
            return Ok(());
        }
        let Some(pipeline) = world.get_resource::<SceneColorPipeline>() else {
            return Ok(());
        };
        let cache = world.resource::<PipelineCache>();
        let Some(render_pipeline) = cache.get_render_pipeline(pipeline.pipeline) else {
            return Ok(());
        };
        let device = world.resource::<RenderDevice>();
        let source = view_target.main_texture_view();
        let epoch = requests.epoch();
        let encoded = output
            .frames
            .encode_current(epoch, request.generation, |destination| {
                let bind = device.create_bind_group(
                    "temporal assembled scene color",
                    &cache.get_bind_group_layout(&pipeline.layout),
                    &BindGroupEntries::single(source),
                );
                let mut pass = context
                    .command_encoder()
                    .begin_render_pass(&RenderPassDescriptor {
                        label: Some("temporal assembled scene color"),
                        color_attachments: &[Some(RenderPassColorAttachment {
                            view: &destination.default_view,
                            resolve_target: None,
                            depth_slice: None,
                            ops: Operations {
                                load: LoadOp::Clear(Default::default()),
                                store: StoreOp::Store,
                            },
                        })],
                        ..default()
                    });
                pass.set_pipeline(render_pipeline);
                pass.set_bind_group(0, &bind, &[]);
                pass.draw(0..3, 0..1);
                drop(pass);
                true
            });
        if encoded {
            debug_assert!(output.current_color(epoch, request.generation).is_some());
        }
        Ok(())
    }
}

/// Private child plugin: registered from [`super::AaPlugin`].
pub(super) struct TemporalSceneColorPlugin;

impl Plugin for TemporalSceneColorPlugin {
    fn build(&self, app: &mut App) {
        let Some(render_app) = app.get_sub_app_mut(RenderApp) else {
            return;
        };
        render_app
            .add_systems(
                Render,
                prepare_scene_color_resources
                    .after(depth::prepare_depth_textures)
                    .in_set(RenderSystems::PrepareBindGroups),
            )
            .add_systems(RenderStartup, init_scene_color_pipeline)
            .add_render_graph_node::<ViewNodeRunner<TemporalSceneColorNode>>(
                Core3d,
                TemporalSceneColorLabel,
            )
            .add_render_graph_edges(
                Core3d,
                (
                    TemporalMotionLabel,
                    TemporalSceneColorLabel,
                    Node3d::EndMainPass,
                ),
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
    let output = world.get::<TemporalSceneColor>(render)?;
    Some(super::gpu_validation::TextureProbe {
        key: output.key(),
        current: output
            .frames
            .current(epoch, generation)
            .map(|texture| texture.texture.clone()),
        previous: output
            .frames
            .previous(epoch, generation)
            .map(|texture| texture.texture.clone()),
    })
}

#[cfg(test)]
pub(super) fn gpu_pipeline_ready(world: &World) -> bool {
    world
        .get_resource::<SceneColorPipeline>()
        .and_then(|pipeline| {
            world
                .get_resource::<PipelineCache>()?
                .get_render_pipeline(pipeline.pipeline)
        })
        .is_some()
}

#[cfg(test)]
pub(super) fn gpu_pipeline_diagnostics(world: &World) -> Vec<String> {
    let Some(pipeline) = world.get_resource::<SceneColorPipeline>() else {
        return vec!["scene-color pipeline: resource absent".into()];
    };
    let cache = world.resource::<PipelineCache>();
    vec![format!(
        "scene-color pipeline: {:?}",
        cache.get_render_pipeline_state(pipeline.pipeline)
    )]
}

#[cfg(test)]
mod tests;
