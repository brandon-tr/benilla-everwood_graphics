//! Reactive-mask publication for scene-color changes caused by transmissive/transparent rendering.
//!
//! The lane copies the opaque scene color after SSAO, then compares it with the color after
//! `MainTransparentPass`. A changed or non-finite RGBA pixel, in a clamped 3x3 neighborhood, marks
//! the center pixel reactive. The mask is a current-frame changed-color rejection signal only:
//! `0` does not prove a rigid surface, correct object velocity, valid history or disocclusion
//! acceptance. Moving opaque/skinned geometry, foliage, animated materials and later atmospheric
//! passes remain outside its coverage.
//!
//! The future resolve must consider both the current and reprojected previous reactive masks so a
//! disappearing effect cannot leave history trails. This task retains one current/previous pair per
//! view, performs no reconstruction and exposes no graphical option. The pair is not color/depth
//! reconstruction history.

use bevy::core_pipeline::core_3d::graph::{Core3d, Node3d};
use bevy::ecs::query::QueryItem;
use bevy::image::ToExtents;
use bevy::prelude::*;
use bevy::render::render_graph::{
    NodeRunError, RenderGraphContext, RenderGraphExt, RenderLabel, ViewNode, ViewNodeRunner,
};
use bevy::render::render_resource::*;
use bevy::render::renderer::{RenderContext, RenderDevice};
use bevy::render::texture::CachedTexture;
use bevy::render::view::ViewTarget;
use bevy::render::{Render, RenderApp, RenderStartup, RenderSystems};

use super::depth::{self, DepthKey, DepthPublication};
use super::frame_pair::TemporalFramePair;

/// One view's opaque snapshot and reactive mask pair. The mask itself is never a prerequisite
/// for producing the mask; both publications are strict consumers for later stages.
#[derive(Component)]
pub(super) struct TemporalReactiveOutput {
    key: DepthKey,
    opaque: CachedTexture,
    mask: TemporalFramePair<CachedTexture>,
    opaque_publication: DepthPublication,
}

impl TemporalReactiveOutput {
    fn new(key: DepthKey, opaque: CachedTexture, masks: [CachedTexture; 2]) -> Self {
        Self {
            key,
            opaque,
            mask: TemporalFramePair::new(key.generation, masks),
            opaque_publication: DepthPublication::new(key.generation),
        }
    }

    pub(super) fn key(&self) -> DepthKey {
        self.key
    }

    /// Strict consumer access to the post-SSAO opaque color for exactly this epoch/generation.
    fn current_opaque(&self, epoch: u64, generation: u64) -> Option<&TextureView> {
        self.opaque_publication
            .is_current(epoch, generation)
            .then_some(&self.opaque.default_view)
    }

    /// Strict consumer access to the current reactive mask for exactly this epoch/generation.
    pub(super) fn current_mask(&self, epoch: u64, generation: u64) -> Option<&TextureView> {
        self.mask
            .current(epoch, generation)
            .map(|mask| &mask.default_view)
    }

    /// The immediately preceding reactive mask, consumed by motion-candidate history.
    pub(super) fn previous_mask(&self, epoch: u64, generation: u64) -> Option<&TextureView> {
        self.mask
            .previous(epoch, generation)
            .map(|mask| &mask.default_view)
    }
}

/// The production mask prerequisite: no mask pass may encode against a missing, stale or
/// wrong-generation opaque snapshot. The first mask still needs no prior mask content.
fn encode_reactive_mask<T>(
    opaque: &DepthPublication,
    mask: &TemporalFramePair<T>,
    epoch: u64,
    generation: u64,
    encode: impl FnOnce(&T) -> bool,
) -> bool {
    if !opaque.is_current(epoch, generation) {
        return false;
    }
    mask.encode_current(epoch, generation, encode)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ReactiveDecision {
    Keep,
    Replace,
    Remove,
}

/// The allocation/retirement decision shared by preparation and the lifecycle tests. A failed
/// source check retires; a matching key reuses; anything else replaces.
fn reactive_decision(
    existing: Option<DepthKey>,
    requested: DepthKey,
    source_valid: bool,
) -> ReactiveDecision {
    if !source_valid {
        ReactiveDecision::Remove
    } else if existing == Some(requested) {
        ReactiveDecision::Keep
    } else {
        ReactiveDecision::Replace
    }
}

/// The snapshot/mask lane only handles ordinary HDR, single-sample, full-target scene color.
pub(super) fn source_matches(view: &ViewTarget, request: &depth::TemporalDepthRequest) -> bool {
    let texture = view.main_texture();
    let size = texture.size();
    let texture_size = UVec2::new(size.width, size.height);
    texture_size == request.size
        && size.depth_or_array_layers == 1
        && view.main_texture_format() == TextureFormat::Rgba16Float
        && texture.format() == TextureFormat::Rgba16Float
        && texture.sample_count() == 1
        && texture.usage().contains(TextureUsages::TEXTURE_BINDING)
}

pub(super) fn reactive_descriptor(
    size: UVec2,
    format: TextureFormat,
    label: &'static str,
) -> TextureDescriptor<'static> {
    TextureDescriptor {
        label: Some(label),
        size: size.to_extents(),
        mip_level_count: 1,
        sample_count: 1,
        dimension: TextureDimension::D2,
        format,
        usage: TextureUsages::RENDER_ATTACHMENT | TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    }
}

/// Prepare allocations for this extraction's requests. Depth publication is deliberately not
/// required: depth has not encoded yet. A matching depth allocation key plus a valid live scene
/// target is enough to own the snapshot and two mask textures.
#[allow(clippy::type_complexity)] // Bevy query tuples read better inline.
fn prepare_reactive_resources(
    mut commands: Commands,
    device: Res<RenderDevice>,
    requests: Res<depth::TemporalDepthRequests>,
    views: Query<(
        Entity,
        &ViewTarget,
        &depth::TemporalDepthTexture,
        Option<&TemporalReactiveOutput>,
    )>,
    outputs: Query<(Entity, &TemporalReactiveOutput)>,
) {
    for (render, request) in requests.iter() {
        let Ok((entity, view, depth_texture, existing)) = views.get(render) else {
            commands.entity(render).remove::<TemporalReactiveOutput>();
            continue;
        };
        let source_valid = depth_texture.key() == request.key() && source_matches(view, request);
        let decision = reactive_decision(
            existing.map(|output| output.key),
            request.key(),
            source_valid,
        );
        match decision {
            ReactiveDecision::Remove => {
                commands.entity(entity).remove::<TemporalReactiveOutput>();
            }
            ReactiveDecision::Keep => {}
            ReactiveDecision::Replace => {
                // 10 bytes/pixel per eligible view: 8 for the Rgba16Float snapshot and 2 for the
                // two R8Unorm masks, excluding allocator overhead. Across both mask lanes this
                // adds 2 bytes/pixel.
                let opaque = device.create_texture(&reactive_descriptor(
                    request.size,
                    TextureFormat::Rgba16Float,
                    "temporal reactive opaque snapshot",
                ));
                let mask_a = device.create_texture(&reactive_descriptor(
                    request.size,
                    TextureFormat::R8Unorm,
                    "temporal reactive mask A",
                ));
                let mask_b = device.create_texture(&reactive_descriptor(
                    request.size,
                    TextureFormat::R8Unorm,
                    "temporal reactive mask B",
                ));
                let opaque = CachedTexture {
                    default_view: opaque.create_view(&TextureViewDescriptor::default()),
                    texture: opaque,
                };
                let mask_a = CachedTexture {
                    default_view: mask_a.create_view(&TextureViewDescriptor::default()),
                    texture: mask_a,
                };
                let mask_b = CachedTexture {
                    default_view: mask_b.create_view(&TextureViewDescriptor::default()),
                    texture: mask_b,
                };
                commands.entity(entity).insert(TemporalReactiveOutput::new(
                    request.key(),
                    opaque,
                    [mask_a, mask_b],
                ));
            }
        }
    }

    for (entity, _) in &outputs {
        if !requests.contains(entity) {
            commands.entity(entity).remove::<TemporalReactiveOutput>();
        }
    }
}

#[derive(Resource)]
struct TemporalReactivePipelines {
    snapshot_layout: BindGroupLayoutDescriptor,
    reactive_layout: BindGroupLayoutDescriptor,
    snapshot: CachedRenderPipelineId,
    reactive: CachedRenderPipelineId,
}

fn texture_layout_entry(binding: u32) -> BindGroupLayoutEntry {
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

fn init_reactive_pipelines(
    mut commands: Commands,
    asset_server: Option<Res<AssetServer>>,
    cache: Res<PipelineCache>,
) {
    let Some(asset_server) = asset_server else {
        return;
    };
    let snapshot_layout =
        BindGroupLayoutDescriptor::new("temporal reactive snapshot", &[texture_layout_entry(0)]);
    let reactive_layout = BindGroupLayoutDescriptor::new(
        "temporal reactive mask",
        &[texture_layout_entry(0), texture_layout_entry(1)],
    );
    let shader: Handle<Shader> =
        asset_server.load("embedded://benilla_world/shaders/temporal_reactive.wgsl");
    let snapshot = cache.queue_render_pipeline(RenderPipelineDescriptor {
        label: Some("temporal reactive snapshot".into()),
        layout: vec![snapshot_layout.clone()],
        vertex: VertexState {
            shader: shader.clone(),
            entry_point: Some("vertex".into()),
            ..default()
        },
        fragment: Some(FragmentState {
            shader: shader.clone(),
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
    let reactive = cache.queue_render_pipeline(RenderPipelineDescriptor {
        label: Some("temporal reactive mask".into()),
        layout: vec![reactive_layout.clone()],
        vertex: VertexState {
            shader: shader.clone(),
            entry_point: Some("vertex".into()),
            ..default()
        },
        fragment: Some(FragmentState {
            shader,
            entry_point: Some("reactive_fragment".into()),
            targets: vec![Some(ColorTargetState {
                format: TextureFormat::R8Unorm,
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
    commands.insert_resource(TemporalReactivePipelines {
        snapshot_layout,
        reactive_layout,
        snapshot,
        reactive,
    });
}

/// The single approved public graph-ordering seam for the app-owned SSAO plugin.
#[derive(Debug, Hash, PartialEq, Eq, Clone, RenderLabel)]
pub struct TemporalOpaqueSnapshotLabel;

#[derive(Debug, Hash, PartialEq, Eq, Clone, RenderLabel)]
pub(super) struct TemporalReactiveLabel;

#[derive(Default)]
struct TemporalOpaqueSnapshotNode;

impl ViewNode for TemporalOpaqueSnapshotNode {
    type ViewQuery = (&'static ViewTarget, &'static TemporalReactiveOutput);

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
        if output.key != request.key() || !source_matches(view_target, request) {
            return Ok(());
        }
        let Some(pipelines) = world.get_resource::<TemporalReactivePipelines>() else {
            return Ok(());
        };
        let cache = world.resource::<PipelineCache>();
        let Some(snapshot_pipeline) = cache.get_render_pipeline(pipelines.snapshot) else {
            return Ok(());
        };
        let source = view_target.main_texture_view();
        let destination = &output.opaque.default_view;
        let device = world.resource::<RenderDevice>();
        let epoch = requests.epoch();
        let encoded = output
            .opaque_publication
            .encode(epoch, request.generation, || {
                let bind = device.create_bind_group(
                    "temporal reactive snapshot",
                    &cache.get_bind_group_layout(&pipelines.snapshot_layout),
                    &BindGroupEntries::single(source),
                );
                let mut pass = context
                    .command_encoder()
                    .begin_render_pass(&RenderPassDescriptor {
                        label: Some("temporal reactive opaque snapshot"),
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
                pass.set_pipeline(snapshot_pipeline);
                pass.set_bind_group(0, &bind, &[]);
                pass.draw(0..3, 0..1);
                drop(pass);
                true
            });
        if encoded {
            debug_assert!(output.current_opaque(epoch, request.generation).is_some());
        }
        Ok(())
    }
}

#[derive(Default)]
struct TemporalReactiveNode;

impl ViewNode for TemporalReactiveNode {
    type ViewQuery = (&'static ViewTarget, &'static TemporalReactiveOutput);

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
        if output.key != request.key() || !source_matches(view_target, request) {
            return Ok(());
        }
        let epoch = requests.epoch();
        let Some(opaque) = output.current_opaque(epoch, request.generation) else {
            return Ok(());
        };
        let Some(pipelines) = world.get_resource::<TemporalReactivePipelines>() else {
            return Ok(());
        };
        let cache = world.resource::<PipelineCache>();
        let Some(reactive_pipeline) = cache.get_render_pipeline(pipelines.reactive) else {
            return Ok(());
        };
        let current = view_target.main_texture_view();
        let device = world.resource::<RenderDevice>();
        let encoded = encode_reactive_mask(
            &output.opaque_publication,
            &output.mask,
            epoch,
            request.generation,
            |destination| {
                let bind = device.create_bind_group(
                    "temporal reactive mask",
                    &cache.get_bind_group_layout(&pipelines.reactive_layout),
                    &BindGroupEntries::sequential((opaque, current)),
                );
                let mut pass = context
                    .command_encoder()
                    .begin_render_pass(&RenderPassDescriptor {
                        label: Some("temporal reactive mask"),
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
                pass.set_pipeline(reactive_pipeline);
                pass.set_bind_group(0, &bind, &[]);
                pass.draw(0..3, 0..1);
                drop(pass);
                true
            },
        );
        if encoded {
            debug_assert!(output.current_mask(epoch, request.generation).is_some());
        }
        Ok(())
    }
}

/// Private child plugin: registered from [`super::AaPlugin`].
pub(super) struct TemporalReactivePlugin;

impl Plugin for TemporalReactivePlugin {
    fn build(&self, app: &mut App) {
        let Some(render_app) = app.get_sub_app_mut(RenderApp) else {
            return;
        };
        render_app
            .add_systems(
                Render,
                prepare_reactive_resources
                    .after(depth::prepare_depth_textures)
                    .in_set(RenderSystems::PrepareBindGroups),
            )
            .add_systems(RenderStartup, init_reactive_pipelines)
            .add_render_graph_node::<ViewNodeRunner<TemporalOpaqueSnapshotNode>>(
                Core3d,
                TemporalOpaqueSnapshotLabel,
            )
            .add_render_graph_node::<ViewNodeRunner<TemporalReactiveNode>>(
                Core3d,
                TemporalReactiveLabel,
            )
            .add_render_graph_edges(
                Core3d,
                (
                    Node3d::MainOpaquePass,
                    TemporalOpaqueSnapshotLabel,
                    Node3d::MainTransmissivePass,
                ),
            )
            .add_render_graph_edges(
                Core3d,
                (
                    Node3d::MainTransparentPass,
                    TemporalReactiveLabel,
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
    let output = world.get::<TemporalReactiveOutput>(render)?;
    Some(super::gpu_validation::TextureProbe {
        key: output.key(),
        current: output
            .mask
            .current(epoch, generation)
            .map(|texture| texture.texture.clone()),
        previous: output
            .mask
            .previous(epoch, generation)
            .map(|texture| texture.texture.clone()),
    })
}

#[cfg(test)]
pub(super) fn gpu_pipeline_ready(world: &World) -> bool {
    let Some(pipelines) = world.get_resource::<TemporalReactivePipelines>() else {
        return false;
    };
    let Some(cache) = world.get_resource::<PipelineCache>() else {
        return false;
    };
    cache.get_render_pipeline(pipelines.snapshot).is_some()
        && cache.get_render_pipeline(pipelines.reactive).is_some()
}

#[cfg(test)]
pub(super) fn gpu_pipeline_diagnostics(world: &World) -> Vec<String> {
    let Some(pipelines) = world.get_resource::<TemporalReactivePipelines>() else {
        return vec!["reactive pipelines: resource absent".into()];
    };
    let cache = world.resource::<PipelineCache>();
    vec![
        format!(
            "reactive snapshot pipeline: {:?}",
            cache.get_render_pipeline_state(pipelines.snapshot)
        ),
        format!(
            "reactive mask pipeline: {:?}",
            cache.get_render_pipeline_state(pipelines.reactive)
        ),
    ]
}

#[cfg(test)]
mod tests;
