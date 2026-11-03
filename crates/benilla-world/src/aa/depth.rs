//! Per-view opaque-depth publication for future native TAA.
//!
//! This lane is deliberately dormant unless the *effective* AA mode is temporal. A requested TAA
//! that fell back to MSAA because `AaReadiness` is off issues no request, allocates no texture and
//! encodes no pass. Even when a request is issued, this module never begins or
//! commits [`super::TemporalHistory`]; a depth copy is not a reconstruction.
//!
//! The output owns two full-target, single-sample `R32Float` copies of the world camera's actual
//! reverse-depth raster, written alternately by extraction epoch and made after `MainOpaquePass`
//! and before `MainTransparentPass`. Retained static geometry reaches that boundary through
//! `StaticGxLabel -> MainOpaquePass`, so both opaque paths are covered. Water's own depth copy and
//! SSAO use the same source read-only; this lane neither depends on water quality nor changes their
//! ordering. The second slot retains the immediately preceding capture without another pass/copy.
//!
//! Each slot's publication is a command-encoding stamp, not GPU completion and not valid TAA
//! history. It is published only after its copy pass has been recorded. `current_depth` accepts
//! exactly the extraction epoch/generation, and `previous_depth` advertises the immediately
//! preceding one only when both slots are published for that generation. A skipped node, an
//! unavailable pipeline, a gap or a stale allocation therefore advertises nothing. Stored depth is
//! the scene's actual reverse-Z raster, preserving its coverage. Previous depth still must be
//! compared against expected previous depth before reconstruction can reuse history; this module
//! neither performs that comparison nor creates/commits color reconstruction history.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};

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
use bevy::render::view::{Msaa, ViewDepthTexture};
use bevy::render::{Extract, ExtractSchedule, Render, RenderApp, RenderStartup, RenderSystems};

use super::frame_pair::TemporalFramePair;
use super::native::NativeResolveBridge;
use super::{AaEffective, AaPolicy, TemporalHistory, TemporalTarget, ViewHistory};
use crate::view::WorldCamera;

/// A stable depth target key. Reuse is legal only for the same main view, history generation and
/// physical extent; the render-world entity is the map key and is never substituted for this.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct DepthKey {
    pub(super) main_view: Entity,
    pub(super) generation: u64,
    pub(super) size: UVec2,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct TemporalDepthRequest {
    pub(super) main_view: Entity,
    pub(super) generation: u64,
    pub(super) size: UVec2,
}

impl TemporalDepthRequest {
    pub(super) fn key(self) -> DepthKey {
        DepthKey {
            main_view: self.main_view,
            generation: self.generation,
            size: self.size,
        }
    }
}

/// One extraction's request map, keyed by render entity because a render view entity and its main
/// camera are distinct `Entity` values. The epoch advances once per extraction and the map is
/// cleared/reused, so a view that vanishes this frame cannot leave a request behind.
#[derive(Resource, Default)]
pub(super) struct TemporalDepthRequests {
    epoch: u64,
    by_render: HashMap<Entity, TemporalDepthRequest>,
}

impl TemporalDepthRequests {
    fn begin_epoch(&mut self) -> u64 {
        self.epoch = self.epoch.wrapping_add(1);
        if self.epoch == 0 {
            self.epoch = 1;
        }
        self.by_render.clear();
        self.epoch
    }

    pub(super) fn epoch(&self) -> u64 {
        self.epoch
    }

    fn insert(&mut self, render: Entity, request: TemporalDepthRequest) {
        self.by_render.insert(render, request);
    }

    pub(super) fn get(&self, render: Entity) -> Option<&TemporalDepthRequest> {
        self.by_render.get(&render)
    }

    pub(super) fn contains(&self, render: Entity) -> bool {
        self.by_render.contains_key(&render)
    }

    pub(super) fn iter(&self) -> impl Iterator<Item = (Entity, &TemporalDepthRequest)> {
        self.by_render
            .iter()
            .map(|(&view, request)| (view, request))
    }
}

/// Whether a view is eligible for temporal-depth publication in this first slice.
///
/// Effective temporal only: a requested TAA or upscale that resolved to MSAA is not eligible. The
/// camera must be active, have actual MSAA off, use no custom viewport/sub-view, have a nonzero
/// target and a registered history generation. Unsupported views produce no request.
fn eligible_request(
    main_view: Entity,
    effective: AaEffective,
    camera: &Camera,
    msaa: Msaa,
    target: Option<TemporalTarget>,
    history: Option<&ViewHistory>,
) -> Option<TemporalDepthRequest> {
    let target = target?;
    if target.size() == UVec2::ZERO {
        return None;
    }
    let history = history?;
    if !effective.is_temporal()
        || !camera.is_active
        || msaa != Msaa::Off
        || camera.viewport.is_some()
        || camera.sub_camera_view.is_some()
    {
        return None;
    }
    Some(TemporalDepthRequest {
        main_view,
        generation: history.generation(),
        size: target.size(),
    })
}

/// The encoded-epoch stamp. `0` is never a valid extraction epoch, so a fresh allocation
/// advertises nothing. Release on publish and acquire on read keep the stamp ordered after the
/// recorded copy; it still says only that commands were encoded, not that the GPU completed.
#[derive(Debug)]
pub(super) struct DepthPublication {
    generation: u64,
    encoded_epoch: AtomicU64,
}

impl DepthPublication {
    pub(super) fn new(generation: u64) -> Self {
        Self {
            generation,
            encoded_epoch: AtomicU64::new(0),
        }
    }

    fn publish(&self, epoch: u64) {
        self.encoded_epoch.store(epoch, Ordering::Release);
    }

    pub(super) fn is_current(&self, epoch: u64, generation: u64) -> bool {
        // `0` is the unencoded sentinel, never a valid extraction epoch.
        epoch != 0
            && self.generation == generation
            && self.encoded_epoch.load(Ordering::Acquire) == epoch
    }

    /// Execute one production encode. This is the single protocol the node and CPU tests use.
    ///
    /// Producer eligibility is the caller's request/allocation/source/pipeline checks, not a
    /// previous publication: a fresh allocation and the next extraction epoch must both encode.
    /// `0` is the unencoded sentinel and a different history generation is obsolete; neither runs
    /// the closure. A closure that fails leaves the publication unchanged.
    pub(super) fn encode(
        &self,
        epoch: u64,
        generation: u64,
        encode: impl FnOnce() -> bool,
    ) -> bool {
        if epoch == 0 || self.generation != generation {
            return false;
        }
        if !encode() {
            return false;
        }
        self.publish(epoch);
        true
    }
}

/// The render view's owned output. The textures are never shared between views, and a cached
/// allocation alone never implies valid content. The pair retains current and immediately previous
/// opaque depth; it is not color reconstruction history.
#[derive(Component)]
pub(super) struct TemporalDepthTexture {
    key: DepthKey,
    frames: TemporalFramePair<CachedTexture>,
}

impl TemporalDepthTexture {
    fn new(textures: [CachedTexture; 2], request: TemporalDepthRequest) -> Self {
        Self {
            key: request.key(),
            frames: TemporalFramePair::new(request.generation, textures),
        }
    }

    pub(super) fn key(&self) -> DepthKey {
        self.key
    }

    /// A valid publication for exactly this extraction epoch and history generation.
    pub(super) fn current_depth(&self, epoch: u64, generation: u64) -> Option<&TextureView> {
        self.frames
            .current(epoch, generation)
            .map(|depth| &depth.default_view)
    }

    /// The immediately preceding opaque-depth capture, when both exact epochs were published.
    /// No later disocclusion comparison samples this yet.
    #[allow(dead_code)] // Future disocclusion consumes it; this task must not add a fake consumer.
    pub(super) fn previous_depth(&self, epoch: u64, generation: u64) -> Option<&TextureView> {
        self.frames
            .previous(epoch, generation)
            .map(|depth| &depth.default_view)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum AllocationDecision {
    Keep,
    Replace,
    Remove,
}

/// Reuse is legal only for an identical key and a currently valid source; any invalid source
/// retires the output even when the key matches.
fn allocation_decision(
    existing: Option<DepthKey>,
    requested: DepthKey,
    source_valid: bool,
) -> AllocationDecision {
    if !source_valid {
        AllocationDecision::Remove
    } else if existing == Some(requested) {
        AllocationDecision::Keep
    } else {
        AllocationDecision::Replace
    }
}

fn depth_descriptor(size: UVec2) -> TextureDescriptor<'static> {
    TextureDescriptor {
        label: Some("temporal depth"),
        size: size.to_extents(),
        mip_level_count: 1,
        sample_count: 1,
        dimension: TextureDimension::D2,
        format: TextureFormat::R32Float,
        usage: TextureUsages::RENDER_ATTACHMENT | TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    }
}

/// Main-world eligibility: the depth attachment must be usable as a texture before Bevy prepares
/// the view's depth texture. Existing usage bits are preserved; nothing is removed here.
#[allow(clippy::type_complexity)] // Bevy query tuples read better inline.
fn ensure_temporal_depth_usage(
    policy: Res<AaPolicy>,
    history: Res<TemporalHistory>,
    mut cameras: Query<
        (
            Entity,
            &Camera,
            &Msaa,
            Option<&TemporalTarget>,
            &mut Camera3d,
        ),
        With<WorldCamera>,
    >,
) {
    for (view, camera, msaa, target, mut camera3d) in &mut cameras {
        if eligible_request(
            view,
            policy.effective(),
            camera,
            *msaa,
            target.copied(),
            history.view(view),
        )
        .is_none()
        {
            continue;
        }
        let usage = TextureUsages::from(camera3d.depth_texture_usages);
        if !usage.contains(TextureUsages::TEXTURE_BINDING) {
            camera3d.depth_texture_usages = (usage | TextureUsages::TEXTURE_BINDING).into();
        }
    }
}

/// Extract the frame's eligible requests. The main-world `Entity` identifies the view; the render
/// entity, translated through `RenderEntity`, keys the request and owns the output.
#[allow(clippy::type_complexity)] // Bevy query tuples read better inline.
pub(super) fn extract_depth_requests(
    mut requests: ResMut<TemporalDepthRequests>,
    bridge: Res<NativeResolveBridge>,
    policy: Extract<Res<AaPolicy>>,
    history: Extract<Res<TemporalHistory>>,
    cameras: Extract<
        Query<
            (
                Entity,
                &RenderEntity,
                &Camera,
                &Msaa,
                Option<&TemporalTarget>,
            ),
            With<WorldCamera>,
        >,
    >,
) {
    let epoch = requests.begin_epoch();
    // The CPU history bridge settles a begun token only once a later extraction proves the render
    // frame completed; this stamp is that proof's clock.
    bridge.note_extraction(epoch);
    for (main_view, render_entity, camera, msaa, target) in cameras.iter() {
        if let Some(request) = eligible_request(
            main_view,
            policy.effective(),
            camera,
            *msaa,
            target.copied(),
            history.view(main_view),
        ) {
            bridge.attempt(main_view, epoch, request.generation);
            requests.insert(render_entity.id(), request);
        }
    }
}

/// Allocate or retire this extraction's per-view outputs after Bevy has prepared the source depth.
/// A view with no current request, or with a source that is no longer single-sample, sized or
/// texture-bindable, keeps no output at all.
#[allow(clippy::type_complexity)] // Bevy query tuples read better inline.
pub(super) fn prepare_depth_textures(
    mut commands: Commands,
    device: Res<RenderDevice>,
    requests: Res<TemporalDepthRequests>,
    views: Query<(
        Entity,
        &ViewDepthTexture,
        &Msaa,
        Option<&TemporalDepthTexture>,
    )>,
    outputs: Query<(Entity, &TemporalDepthTexture)>,
) {
    for (render, request) in requests.iter() {
        let Ok((entity, source, msaa, existing)) = views.get(render) else {
            commands.entity(render).remove::<TemporalDepthTexture>();
            continue;
        };
        let source_size = source.texture.size();
        let size = UVec2::new(source_size.width, source_size.height);
        let source_valid = *msaa == Msaa::Off
            && source.texture.sample_count() == 1
            && source
                .texture
                .usage()
                .contains(TextureUsages::TEXTURE_BINDING)
            && size == request.size;
        let decision = allocation_decision(
            existing.map(TemporalDepthTexture::key),
            request.key(),
            source_valid,
        );
        match decision {
            AllocationDecision::Remove => {
                commands.entity(entity).remove::<TemporalDepthTexture>();
            }
            AllocationDecision::Keep => {}
            AllocationDecision::Replace => {
                // Direct allocation, not `TextureCache`: these textures live across frames and are
                // owned by one view, while the cache reuses transient textures between views.
                // The pair alternates by extraction epoch, so 8 bytes/pixel are retained instead
                // of the previous 4, excluding allocator overhead.
                let texture_a = device.create_texture(&depth_descriptor(request.size));
                let texture_b = device.create_texture(&depth_descriptor(request.size));
                let texture_a = CachedTexture {
                    default_view: texture_a.create_view(&TextureViewDescriptor::default()),
                    texture: texture_a,
                };
                let texture_b = CachedTexture {
                    default_view: texture_b.create_view(&TextureViewDescriptor::default()),
                    texture: texture_b,
                };
                commands
                    .entity(entity)
                    .insert(TemporalDepthTexture::new([texture_a, texture_b], *request));
            }
        }
    }

    for (entity, _) in &outputs {
        if !requests.contains(entity) {
            commands.entity(entity).remove::<TemporalDepthTexture>();
        }
    }
}

#[derive(Resource)]
struct TemporalDepthPipeline {
    layout: BindGroupLayoutDescriptor,
    pipeline: CachedRenderPipelineId,
}

fn init_depth_pipeline(
    mut commands: Commands,
    asset_server: Option<Res<AssetServer>>,
    cache: Res<PipelineCache>,
) {
    // A render app without its asset machinery has no embedded shader to load; the node then
    // advertises nothing until a real pipeline exists.
    let Some(asset_server) = asset_server else {
        return;
    };
    let layout = BindGroupLayoutDescriptor::new(
        "temporal depth source",
        &[BindGroupLayoutEntry {
            binding: 0,
            visibility: ShaderStages::FRAGMENT,
            ty: BindingType::Texture {
                sample_type: TextureSampleType::Depth,
                view_dimension: TextureViewDimension::D2,
                multisampled: false,
            },
            count: None,
        }],
    );
    let shader: Handle<Shader> =
        asset_server.load("embedded://benilla_world/shaders/temporal_depth.wgsl");
    let pipeline = cache.queue_render_pipeline(RenderPipelineDescriptor {
        label: Some("temporal depth copy".into()),
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
                format: TextureFormat::R32Float,
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
    commands.insert_resource(TemporalDepthPipeline { layout, pipeline });
}

#[derive(Debug, Hash, PartialEq, Eq, Clone, RenderLabel)]
pub(super) struct TemporalDepthLabel;

#[derive(Default)]
struct TemporalDepthNode;

impl ViewNode for TemporalDepthNode {
    type ViewQuery = (&'static ViewDepthTexture, &'static TemporalDepthTexture);

    fn run<'w>(
        &self,
        graph: &mut RenderGraphContext,
        context: &mut RenderContext<'w>,
        (source, output): QueryItem<'w, '_, Self::ViewQuery>,
        world: &'w World,
    ) -> Result<(), NodeRunError> {
        let requests = world.resource::<TemporalDepthRequests>();
        let Some(request) = requests.get(graph.view_entity()) else {
            return Ok(());
        };
        if output.key() != request.key() {
            return Ok(());
        }
        let source_size = source.texture.size();
        if UVec2::new(source_size.width, source_size.height) != request.size
            || source.texture.sample_count() != 1
            || !source
                .texture
                .usage()
                .contains(TextureUsages::TEXTURE_BINDING)
        {
            return Ok(());
        }
        // The selected pair slot is the producer's write attachment. `current_depth` is the
        // consumer protocol and must not gate the producer that creates its first publication.
        let Some(pipeline) = world.get_resource::<TemporalDepthPipeline>() else {
            return Ok(());
        };
        let cache = world.resource::<PipelineCache>();
        let Some(render_pipeline) = cache.get_render_pipeline(pipeline.pipeline) else {
            return Ok(());
        };
        let device = world.resource::<RenderDevice>();
        let epoch = requests.epoch();
        let encoded = output
            .frames
            .encode_current(epoch, request.generation, |destination| {
                let bind = device.create_bind_group(
                    "temporal depth source",
                    &cache.get_bind_group_layout(&pipeline.layout),
                    &BindGroupEntries::single(source.view()),
                );
                let mut pass = context
                    .command_encoder()
                    .begin_render_pass(&RenderPassDescriptor {
                        label: Some("temporal opaque depth capture"),
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
            // The shared producer/consumer invariant; `current_depth` remains the consumer view.
            debug_assert!(output.current_depth(epoch, request.generation).is_some());
        }
        Ok(())
    }
}

/// Private child plugin: registered from [`super::AaPlugin`].
pub(super) struct TemporalDepthPlugin;

impl Plugin for TemporalDepthPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(
            Last,
            ensure_temporal_depth_usage.after(super::resolve_policy),
        );
        let Some(render_app) = app.get_sub_app_mut(RenderApp) else {
            return;
        };
        render_app
            .init_resource::<TemporalDepthRequests>()
            .add_systems(ExtractSchedule, extract_depth_requests)
            .add_systems(RenderStartup, init_depth_pipeline)
            .add_systems(
                Render,
                prepare_depth_textures.in_set(RenderSystems::PrepareBindGroups),
            )
            .add_render_graph_node::<ViewNodeRunner<TemporalDepthNode>>(Core3d, TemporalDepthLabel)
            .add_render_graph_edges(
                Core3d,
                (
                    Node3d::MainOpaquePass,
                    TemporalDepthLabel,
                    Node3d::MainTransparentPass,
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
    let output = world.get::<TemporalDepthTexture>(render)?;
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
        .get_resource::<TemporalDepthPipeline>()
        .and_then(|pipeline| {
            world
                .get_resource::<PipelineCache>()?
                .get_render_pipeline(pipeline.pipeline)
        })
        .is_some()
}

#[cfg(test)]
pub(super) fn gpu_pipeline_diagnostics(world: &World) -> Vec<String> {
    let Some(pipeline) = world.get_resource::<TemporalDepthPipeline>() else {
        return vec!["depth pipeline: resource absent".into()];
    };
    let cache = world.resource::<PipelineCache>();
    vec![format!(
        "depth pipeline: {:?}",
        cache.get_render_pipeline_state(pipeline.pipeline)
    )]
}

#[cfg(test)]
mod tests;
