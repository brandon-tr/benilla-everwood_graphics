//! Test-only reference fixture for stock Bevy 0.18.1 temporal anti-aliasing.
//!
//! This is isolated from Everwood's AA producers and does not enable production TAA. It exists
//! only to prove the unmodified Bevy node executes and reconstructs synthetic geometry.
//!
//! Bevy's TAA history textures and cached pipeline IDs are private. Execution is therefore proven
//! by observing the actual before/after `ViewTarget` texture flip, extracted reset state, jitter,
//! and prepass attachments; this test does not inspect private history or pipeline fields.

use std::sync::Mutex;
use std::time::{Duration, Instant};

use bevy::anti_alias::taa::TemporalAntiAliasing;
use bevy::app::PluginsState;
use bevy::camera::{CameraOutputMode, PerspectiveProjection, RenderTarget};
use bevy::core_pipeline::core_3d::graph::{Core3d, Node3d};
use bevy::core_pipeline::prepass::{DepthPrepass, MotionVectorPrepass, ViewPrepassTextures};
use bevy::core_pipeline::tonemapping::Tonemapping;
use bevy::ecs::query::QueryItem;
use bevy::image::Image;
use bevy::prelude::*;
use bevy::render::camera::{MipBias, TemporalJitter};
use bevy::render::pipelined_rendering::PipelinedRenderingPlugin;
use bevy::render::render_graph::{
    NodeRunError, RenderGraphContext, RenderGraphExt, RenderLabel, ViewNode, ViewNodeRunner,
};
use bevy::render::render_resource::*;
use bevy::render::renderer::RenderAdapterInfo;
use bevy::render::view::{Hdr, ViewTarget};
use bevy::render::RenderApp;
use bevy::shader::Shader;
use bevy::time::TimeUpdateStrategy;
use bevy::window::{ExitCondition, WindowPlugin};
use bevy::winit::WinitPlugin;

use super::gpu_validation::{read_texture_pixels, SAMPLE_WGSL};

pub(super) const SIZE: UVec2 = UVec2::new(64, 64);
const FRAME_SECONDS: f64 = 1.0 / 60.0;

#[derive(Clone, Debug)]
pub(super) struct ObservedCapture {
    pub(super) view: Entity,
    pub(super) frame: u64,
    pub(super) texture: Texture,
    pub(super) reset: bool,
    pub(super) taa_present: bool,
    pub(super) jitter: Option<Vec2>,
    pub(super) depth: Option<(TextureFormat, Extent3d)>,
    pub(super) motion: Option<(TextureFormat, Extent3d, Texture)>,
}

#[derive(Default)]
struct Captures {
    sequence: u64,
    before: Option<ObservedCapture>,
    after: Option<ObservedCapture>,
}

#[derive(Resource, Default)]
struct CaptureStore {
    captures: Mutex<Captures>,
}

impl CaptureStore {
    fn clear(&self) {
        let mut captures = self.captures.lock().unwrap();
        captures.before = None;
        captures.after = None;
    }

    fn snapshot(&self) -> (Option<ObservedCapture>, Option<ObservedCapture>) {
        let captures = self.captures.lock().unwrap();
        (captures.before.clone(), captures.after.clone())
    }
}

fn observe(
    world: &World,
    graph: &RenderGraphContext,
    view_target: &ViewTarget,
    taa: Option<&TemporalAntiAliasing>,
    jitter: Option<&TemporalJitter>,
    prepass: Option<&ViewPrepassTextures>,
    before: bool,
) {
    let store = world.resource::<CaptureStore>();
    let mut captures = store.captures.lock().unwrap();
    let frame = if before {
        captures.sequence = captures.sequence.wrapping_add(1);
        captures.sequence
    } else {
        captures.sequence
    };
    let depth = prepass.and_then(|prepass| {
        prepass
            .depth
            .as_ref()
            .map(|depth| (depth.texture.texture.format(), depth.texture.texture.size()))
    });
    let motion = prepass.and_then(|prepass| {
        prepass.motion_vectors.as_ref().map(|motion| {
            (
                motion.texture.texture.format(),
                motion.texture.texture.size(),
                motion.texture.texture.clone(),
            )
        })
    });
    let capture = ObservedCapture {
        view: graph.view_entity(),
        frame,
        texture: view_target.main_texture().clone(),
        reset: taa.is_some_and(|taa| taa.reset),
        taa_present: taa.is_some(),
        jitter: jitter.map(|jitter| jitter.offset),
        depth,
        motion,
    };
    if before {
        captures.before = Some(capture);
    } else {
        captures.after = Some(capture);
    }
}

#[derive(Default)]
struct ReferenceObserver<const BEFORE: bool>;

impl<const BEFORE: bool> ViewNode for ReferenceObserver<BEFORE> {
    type ViewQuery = (
        &'static ViewTarget,
        Option<&'static TemporalAntiAliasing>,
        Option<&'static TemporalJitter>,
        Option<&'static ViewPrepassTextures>,
    );

    fn run<'w>(
        &self,
        graph: &mut RenderGraphContext,
        _context: &mut bevy::render::renderer::RenderContext<'w>,
        (view_target, taa, jitter, prepass): QueryItem<'w, '_, Self::ViewQuery>,
        world: &'w World,
    ) -> Result<(), NodeRunError> {
        observe(world, graph, view_target, taa, jitter, prepass, BEFORE);
        Ok(())
    }
}

#[derive(Debug, Hash, PartialEq, Eq, Clone, RenderLabel)]
struct BeforeReferenceTaa;

#[derive(Debug, Hash, PartialEq, Eq, Clone, RenderLabel)]
struct AfterReferenceTaa;

struct ReferenceCapturePlugin;

impl Plugin for ReferenceCapturePlugin {
    fn build(&self, app: &mut App) {
        let Some(render_app) = app.get_sub_app_mut(RenderApp) else {
            return;
        };
        render_app
            .insert_resource(CaptureStore::default())
            .add_render_graph_node::<ViewNodeRunner<ReferenceObserver<true>>>(
                Core3d,
                BeforeReferenceTaa,
            )
            .add_render_graph_node::<ViewNodeRunner<ReferenceObserver<false>>>(
                Core3d,
                AfterReferenceTaa,
            )
            .add_render_graph_edges(
                Core3d,
                (Node3d::MotionBlur, BeforeReferenceTaa, Node3d::Taa),
            )
            .add_render_graph_edges(Core3d, (Node3d::Taa, AfterReferenceTaa, Node3d::Bloom));
    }
}

pub(super) struct BevyReferenceFixture {
    pub(super) app: App,
    pub(super) camera: Entity,
    pub(super) sampling_layout: BindGroupLayoutDescriptor,
    pub(super) sampling_pipeline: CachedRenderPipelineId,
    pub(super) stage_deadline: Option<Instant>,
    pub(super) stage_evidence: String,
    hold_reset: bool,
}

impl BevyReferenceFixture {
    pub(super) fn new() -> Self {
        Self::new_with_setup(|_| {})
    }

    pub(super) fn new_with_setup(setup: impl FnOnce(&mut App)) -> Self {
        let mut app = App::new();
        app.add_plugins(
            DefaultPlugins
                .set(WindowPlugin {
                    primary_window: None,
                    exit_condition: ExitCondition::DontExit,
                    ..default()
                })
                .disable::<WinitPlugin>()
                .disable::<PipelinedRenderingPlugin>(),
        )
        .insert_resource(TimeUpdateStrategy::ManualDuration(Duration::from_secs_f64(
            FRAME_SECONDS,
        )))
        .insert_resource(ClearColor(Color::BLACK))
        .add_plugins(ReferenceCapturePlugin);

        setup(&mut app);

        let target =
            app.world_mut()
                .resource_mut::<Assets<Image>>()
                .add(Image::new_target_texture(
                    SIZE.x,
                    SIZE.y,
                    TextureFormat::Rgba16Float,
                    None,
                ));
        let camera = app
            .world_mut()
            .spawn((
                Camera3d::default(),
                Camera {
                    output_mode: CameraOutputMode::Skip,
                    ..default()
                },
                Projection::Perspective(PerspectiveProjection {
                    fov: 60.0f32.to_radians(),
                    ..default()
                }),
                RenderTarget::Image(target.into()),
                Hdr,
                Tonemapping::None,
                Msaa::Off,
                TemporalAntiAliasing::default(),
                Transform::from_xyz(0.0, 0.0, 5.0).looking_at(Vec3::ZERO, Vec3::Y),
            ))
            .id();

        let mesh = app
            .world_mut()
            .resource_mut::<Assets<Mesh>>()
            .add(Rectangle::new(3.0, 3.0));
        let material = app
            .world_mut()
            .resource_mut::<Assets<StandardMaterial>>()
            .add(StandardMaterial {
                base_color: Color::srgb(0.2, 0.6, 0.9),
                unlit: true,
                cull_mode: None,
                ..default()
            });
        app.world_mut().spawn((
            Mesh3d(mesh),
            MeshMaterial3d(material),
            Transform::from_rotation(Quat::from_rotation_z(0.23)),
        ));

        let sampling_shader = app
            .world_mut()
            .resource_mut::<Assets<Shader>>()
            .add(Shader::from_wgsl(SAMPLE_WGSL, "bevy_reference_sample.wgsl"));
        let sampling_layout = BindGroupLayoutDescriptor::new(
            "bevy reference sampling",
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

        let readiness_deadline = Instant::now() + Duration::from_secs(30);
        while app.plugins_state() == PluginsState::Adding {
            if Instant::now() >= readiness_deadline {
                panic!("timed out waiting for Bevy reference plugin/renderer readiness");
            }
            bevy::tasks::tick_global_task_pools_on_main_thread();
            std::thread::sleep(Duration::from_millis(1));
        }
        app.finish();
        app.cleanup();

        let sampling_pipeline = app
            .get_sub_app_mut(RenderApp)
            .expect("render app")
            .world_mut()
            .resource_mut::<PipelineCache>()
            .queue_render_pipeline(RenderPipelineDescriptor {
                label: Some("bevy reference sampling".into()),
                layout: vec![sampling_layout.clone()],
                vertex: VertexState {
                    shader: sampling_shader.clone(),
                    entry_point: Some("vertex".into()),
                    ..default()
                },
                fragment: Some(FragmentState {
                    shader: sampling_shader.clone(),
                    entry_point: Some("sample".into()),
                    targets: vec![Some(ColorTargetState {
                        format: TextureFormat::Rgba32Float,
                        blend: None,
                        write_mask: ColorWrites::ALL,
                    })],
                    ..default()
                }),
                multisample: MultisampleState {
                    count: 1,
                    ..default()
                },
                ..default()
            });

        Self {
            app,
            camera,
            sampling_layout,
            sampling_pipeline,
            stage_deadline: None,
            stage_evidence: String::new(),
            hold_reset: true,
        }
    }

    pub(super) fn update(&mut self) {
        self.clear_captures();
        if self.hold_reset {
            if let Some(mut taa) = self
                .app
                .world_mut()
                .get_mut::<TemporalAntiAliasing>(self.camera)
            {
                taa.reset = true;
            }
        }
        self.app.update();
    }

    pub(super) fn render_world(&self) -> &World {
        self.app.get_sub_app(RenderApp).expect("render app").world()
    }

    pub(super) fn sampling_ready(&self) -> bool {
        self.render_world()
            .resource::<PipelineCache>()
            .get_render_pipeline(self.sampling_pipeline)
            .is_some()
    }

    pub(super) fn captures(&self) -> (Option<ObservedCapture>, Option<ObservedCapture>) {
        self.render_world().resource::<CaptureStore>().snapshot()
    }

    pub(super) fn clear_captures(&self) {
        self.render_world().resource::<CaptureStore>().clear();
    }

    pub(super) fn read_texture(&self, texture: &Texture) -> Vec<[f32; 4]> {
        read_texture_pixels(
            self.render_world(),
            &self.sampling_layout,
            self.sampling_pipeline,
            self.stage_deadline,
            texture,
        )
    }

    pub(super) fn print_adapter(&self) {
        let info = self.app.world().resource::<RenderAdapterInfo>();
        println!("Bevy reference GPU adapter info: {:?}", **info);
    }

    pub(super) fn request_reset(&mut self) {
        self.hold_reset = true;
        if let Some(mut taa) = self
            .app
            .world_mut()
            .get_mut::<TemporalAntiAliasing>(self.camera)
        {
            taa.reset = true;
        }
    }

    pub(super) fn release_reset(&mut self) {
        self.hold_reset = false;
        if let Some(mut taa) = self
            .app
            .world_mut()
            .get_mut::<TemporalAntiAliasing>(self.camera)
        {
            taa.reset = false;
        }
    }

    pub(super) fn camera_cut(&mut self, x: f32) {
        self.app
            .world_mut()
            .entity_mut(self.camera)
            .insert(Transform::from_xyz(x, 0.0, 5.0).looking_at(Vec3::new(x, 0.0, 0.0), Vec3::Y));
        self.request_reset();
    }

    pub(super) fn disable_taa(&mut self) {
        let mut entity = self.app.world_mut().entity_mut(self.camera);
        entity.remove::<TemporalAntiAliasing>();
        entity.remove::<TemporalJitter>();
        entity.remove::<MipBias>();
        entity.remove::<DepthPrepass>();
        entity.remove::<MotionVectorPrepass>();
        self.hold_reset = false;
    }

    pub(super) fn render_entity(&self) -> Option<Entity> {
        self.app
            .world()
            .get::<bevy::render::sync_world::RenderEntity>(self.camera)
            .map(|entity| entity.id())
    }
}

pub(super) fn wait_reference_stage<T>(
    fixture: &mut BevyReferenceFixture,
    stage: &str,
    mut check: impl FnMut(&mut BevyReferenceFixture) -> Option<T>,
) -> T {
    const MAX_UPDATES: u32 = 240;
    let started = Instant::now();
    let deadline = started + Duration::from_secs(30);
    fixture.stage_deadline = Some(deadline);
    fixture.stage_evidence.clear();
    let mut updates = 0;
    loop {
        if updates >= MAX_UPDATES || Instant::now() >= deadline {
            fixture.stage_deadline = None;
            let (before, after) = fixture.captures();
            panic!(
                "timed out in {stage}: updates={updates}, elapsed={:?}, sampling={}, before={:?}, after={:?}, evidence={}",
                started.elapsed(),
                fixture.sampling_ready(),
                before.map(|capture| (capture.frame, capture.view, capture.reset, capture.taa_present)),
                after.map(|capture| (capture.frame, capture.view, capture.reset, capture.taa_present)),
                fixture.stage_evidence,
            );
        }
        fixture.update();
        updates += 1;
        if let Some(value) = check(fixture) {
            fixture.stage_deadline = None;
            return value;
        }
    }
}

#[cfg(test)]
mod terrain;
#[cfg(test)]
mod tests;

#[cfg(test)]
mod early_taa;
