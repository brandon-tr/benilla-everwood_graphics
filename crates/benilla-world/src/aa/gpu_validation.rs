//! Test-only GPU fixture and readback helpers.
//!
//! Nothing here runs at module load. The ignored test constructs the offscreen device, renders
//! through the real AA plugins, and samples real production output textures through a test-owned
//! staging pass. Production textures keep their existing usages.

use std::sync::mpsc;
use std::time::{Duration, Instant};

use benilla_assets::materials::WowModelMaterial;
use bevy::app::PluginsState;
use bevy::camera::{CameraOutputMode, PerspectiveProjection, RenderTarget};
use bevy::core_pipeline::tonemapping::Tonemapping;
use bevy::image::Image;
use bevy::prelude::*;
use bevy::render::pipelined_rendering::PipelinedRenderingPlugin;
use bevy::render::render_resource::*;
use bevy::render::renderer::{RenderAdapterInfo, RenderDevice, RenderQueue};
use bevy::render::view::{Hdr, ViewTarget};
use bevy::render::RenderApp;
use bevy::shader::Shader;
use bevy::time::TimeUpdateStrategy;
use bevy::window::{ExitCondition, WindowPlugin};
use bevy::winit::WinitPlugin;

use super::depth::{self, DepthKey};
use super::motion;
use super::reactive;
use super::rejection;
use super::scene_color;
use super::{
    AaInvalidation, AaMode, AaPlugin, AaPolicy, AaReadiness, TemporalHistory, TemporalTarget,
};
use crate::lighting::MonkeyFrame;
use crate::schedule::WorldLive;
use crate::shaders;
use crate::view::{MsaaSetting, Viewer, WorldCamera};
use crate::world_map::MapChange;

pub(super) const SIZE: UVec2 = UVec2::new(64, 64);
pub(super) const SAMPLE_WGSL: &str = r#"
@group(0) @binding(0) var source: texture_2d<f32>;

@vertex
fn vertex(@builtin(vertex_index) i: u32) -> @builtin(position) vec4<f32> {
    let x = f32((i << 1u) & 2u);
    let y = f32(i & 2u);
    return vec4<f32>(x * 2.0 - 1.0, y * 2.0 - 1.0, 0.0, 1.0);
}

@fragment
fn sample(@builtin(position) p: vec4<f32>) -> @location(0) vec4<f32> {
    return textureLoad(source, vec2<i32>(p.xy), 0);
}
"#;

#[derive(Clone)]
pub(super) struct TextureProbe {
    pub(super) key: DepthKey,
    pub(super) current: Option<Texture>,
    pub(super) previous: Option<Texture>,
}

pub(super) struct Probes {
    pub(super) depth: TextureProbe,
    pub(super) motion: TextureProbe,
    pub(super) reactive: TextureProbe,
    pub(super) rejection: TextureProbe,
    pub(super) scene: TextureProbe,
}

pub(super) struct ViewSlot {
    pub(super) render: Entity,
    pub(super) key: DepthKey,
    pub(super) epoch: u64,
}

pub(super) struct Fixture {
    pub(super) app: App,
    pub(super) camera: Entity,
    pub(super) size: UVec2,
    pub(super) sampling_layout: BindGroupLayoutDescriptor,
    pub(super) sampling_pipeline: CachedRenderPipelineId,
    pub(super) stage_deadline: Option<Instant>,
    pub(super) warmup_evidence: String,
}

fn target_image(size: UVec2) -> Image {
    let mut image = Image::new_target_texture(size.x, size.y, TextureFormat::Rgba16Float, None);
    image.texture_descriptor.usage |= TextureUsages::COPY_SRC;
    image
}

fn camera_bundle(target: Handle<Image>, size: UVec2, transform: Transform) -> impl Bundle {
    (
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
        WorldCamera,
        TemporalTarget::new(size, 1.0),
        transform,
    )
}

impl Fixture {
    pub(super) fn new(size: UVec2) -> Self {
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
        );
        app.insert_resource(MsaaSetting { samples: 1 })
            .init_resource::<Viewer>()
            .init_resource::<WorldLive>()
            .add_message::<MapChange>()
            .insert_resource(TimeUpdateStrategy::ManualDuration(Duration::from_secs_f64(
                1.0 / 60.0,
            )))
            .insert_resource(Assets::<WowModelMaterial>::default());
        shaders::plugin(&mut app);
        app.add_plugins(AaPlugin);
        app.world_mut().resource_mut::<AaReadiness>().taa = true;
        app.world_mut()
            .resource_mut::<AaPolicy>()
            .request(AaMode::Taa);

        let target = app
            .world_mut()
            .resource_mut::<Assets<Image>>()
            .add(target_image(size));
        let camera = app
            .world_mut()
            .spawn(camera_bundle(
                target,
                size,
                Transform::from_xyz(0.0, 0.0, 5.0).looking_at(Vec3::ZERO, Vec3::Y),
            ))
            .id();

        let mesh = app
            .world_mut()
            .resource_mut::<Assets<Mesh>>()
            .add(Rectangle::new(8.0, 8.0));
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
            Transform::from_xyz(0.0, 0.0, 0.0),
        ));
        app.world_mut().insert_resource(ClearColor(Color::BLACK));

        let sampling_shader = app
            .world_mut()
            .resource_mut::<Assets<Shader>>()
            .add(Shader::from_wgsl(SAMPLE_WGSL, "gpu_validation_sample.wgsl"));
        let sampling_layout = BindGroupLayoutDescriptor::new(
            "gpu validation sampling",
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

        // Plugin readiness is part of RenderPlugin initialization: waiting for it before finish
        // ensures FutureRenderResources has produced the device, queue and PipelineCache.
        let readiness_deadline = Instant::now() + Duration::from_secs(30);
        while app.plugins_state() == PluginsState::Adding {
            if Instant::now() >= readiness_deadline {
                panic!("timed out waiting for plugin/renderer readiness");
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
                label: Some("gpu validation sampling".into()),
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
            size,
            sampling_layout,
            sampling_pipeline,
            stage_deadline: None,
            warmup_evidence: String::new(),
        }
    }

    pub(super) fn update(&mut self) {
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

    pub(super) fn slot(&self, main_view: Entity) -> Option<ViewSlot> {
        let requests = self
            .render_world()
            .resource::<depth::TemporalDepthRequests>();
        requests
            .iter()
            .find(|(_, request)| request.key().main_view == main_view)
            .map(|(render, request)| ViewSlot {
                render,
                key: request.key(),
                epoch: requests.epoch(),
            })
    }

    pub(super) fn probes(&self, slot: &ViewSlot) -> Option<Probes> {
        let world = self.render_world();
        Some(Probes {
            depth: depth::gpu_probe(world, slot.render, slot.epoch, slot.key.generation)?,
            motion: motion::gpu_probe(world, slot.render, slot.epoch, slot.key.generation)?,
            reactive: reactive::gpu_probe(world, slot.render, slot.epoch, slot.key.generation)?,
            rejection: rejection::gpu_probe(world, slot.render, slot.epoch, slot.key.generation)?,
            scene: scene_color::gpu_probe(world, slot.render, slot.epoch, slot.key.generation)?,
        })
    }

    pub(super) fn add_transparent(&mut self) -> Entity {
        let mesh = self
            .app
            .world_mut()
            .resource_mut::<Assets<Mesh>>()
            .add(Rectangle::new(3.0, 3.0));
        let material = self
            .app
            .world_mut()
            .resource_mut::<Assets<StandardMaterial>>()
            .add(StandardMaterial {
                base_color: Color::srgba(1.0, 0.2, 0.1, 0.5),
                alpha_mode: AlphaMode::Blend,
                unlit: true,
                cull_mode: None,
                ..default()
            });
        self.app
            .world_mut()
            .spawn((
                Mesh3d(mesh),
                MeshMaterial3d(material),
                Transform::from_xyz(0.0, 0.0, 1.0),
            ))
            .id()
    }

    pub(super) fn add_wind_metadata(&mut self) {
        let frame = MonkeyFrame {
            wind_dir: [1.0, 0.0],
            ..Default::default()
        };
        self.app
            .world_mut()
            .insert_resource(crate::static_gx::wind_source(1));
        self.app.world_mut().insert_resource(frame);
    }

    pub(super) fn resize(&mut self, size: UVec2) {
        let target = self
            .app
            .world_mut()
            .resource_mut::<Assets<Image>>()
            .add(target_image(size));
        let mut camera = self.app.world_mut().entity_mut(self.camera);
        camera.insert((
            RenderTarget::Image(target.clone().into()),
            TemporalTarget::new(size, 1.0),
        ));
        self.size = size;
    }

    pub(super) fn add_second_camera(&mut self, size: UVec2) -> (Entity, Handle<Image>) {
        let target = self
            .app
            .world_mut()
            .resource_mut::<Assets<Image>>()
            .add(target_image(size));
        let camera = self
            .app
            .world_mut()
            .spawn(camera_bundle(
                target.clone(),
                size,
                Transform::from_xyz(20.0, 0.0, 5.0).looking_at(Vec3::new(20.0, 0.0, 0.0), Vec3::Y),
            ))
            .id();
        (camera, target)
    }

    pub(super) fn read_view_source(&self, render: Entity) -> Vec<[f32; 4]> {
        let world = self.render_world();
        let view = world.get::<ViewTarget>(render).expect("render ViewTarget");
        self.read_texture(view.main_texture())
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
        println!("GPU adapter info: {:?}", **info);
    }

    pub(super) fn request_off(&mut self) {
        self.app
            .world_mut()
            .resource_mut::<AaPolicy>()
            .request(AaMode::Off);
    }

    pub(super) fn request_taa(&mut self) {
        self.app
            .world_mut()
            .resource_mut::<AaPolicy>()
            .request(AaMode::Taa);
    }

    pub(super) fn invalidate(&mut self, reason: AaInvalidation) {
        self.app
            .world_mut()
            .resource_mut::<TemporalHistory>()
            .invalidate(self.camera, reason);
    }
}

pub(super) fn read_texture_pixels(
    world: &World,
    layout: &BindGroupLayoutDescriptor,
    pipeline: CachedRenderPipelineId,
    stage_deadline: Option<Instant>,
    texture: &Texture,
) -> Vec<[f32; 4]> {
    let device = world.resource::<RenderDevice>();
    let queue = world.resource::<RenderQueue>();
    let cache = world.resource::<PipelineCache>();
    let size = texture.size();
    let width = size.width;
    let height = size.height;
    let readback_deadline = Instant::now() + Duration::from_secs(10);
    let deadline = stage_deadline.map_or(readback_deadline, |stage| stage.min(readback_deadline));
    let remaining = || {
        let remaining = deadline.saturating_duration_since(Instant::now());
        assert!(
            !remaining.is_zero(),
            "readback deadline expired before GPU completion"
        );
        remaining
    };
    let render_pipeline = cache
        .get_render_pipeline(pipeline)
        .expect("sampling pipeline ready");
    let bind_layout = cache.get_bind_group_layout(layout);
    let staged = device.create_texture(&TextureDescriptor {
        label: Some("gpu validation staging"),
        size,
        mip_level_count: 1,
        sample_count: 1,
        dimension: TextureDimension::D2,
        format: TextureFormat::Rgba32Float,
        usage: TextureUsages::RENDER_ATTACHMENT | TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    let staged_view = staged.create_view(&TextureViewDescriptor::default());
    let source_view = texture.create_view(&TextureViewDescriptor::default());
    let bind = device.create_bind_group(
        "gpu validation sampling",
        &bind_layout,
        &BindGroupEntries::single(&source_view),
    );
    let padded = RenderDevice::align_copy_bytes_per_row(width as usize * 16);
    let buffer = device.create_buffer(&BufferDescriptor {
        label: Some("gpu validation readback"),
        size: (padded * height as usize) as u64,
        usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
        label: Some("gpu validation readback"),
    });
    {
        let mut pass = encoder.begin_render_pass(&RenderPassDescriptor {
            label: Some("gpu validation sampling"),
            color_attachments: &[Some(RenderPassColorAttachment {
                view: &staged_view,
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
    }
    encoder.copy_texture_to_buffer(
        TexelCopyTextureInfo {
            texture: &staged,
            mip_level: 0,
            origin: Origin3d::ZERO,
            aspect: TextureAspect::All,
        },
        TexelCopyBufferInfo {
            buffer: &buffer,
            layout: TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(padded as u32),
                rows_per_image: Some(height),
            },
        },
        Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
    );
    let submission = queue.submit([encoder.finish()]);
    device
        .wgpu_device()
        .poll(PollType::Wait {
            submission_index: Some(submission),
            timeout: Some(remaining()),
        })
        .expect("GPU wait");

    let (sender, receiver) = mpsc::channel();
    buffer.slice(..).map_async(MapMode::Read, move |result| {
        let _ = sender.send(result);
    });
    device
        .wgpu_device()
        .poll(PollType::Wait {
            submission_index: None,
            timeout: Some(remaining()),
        })
        .expect("GPU map wait");
    receiver
        .recv_timeout(remaining())
        .expect("map callback")
        .expect("buffer map");

    let mut pixels = Vec::with_capacity((width * height) as usize);
    {
        let data = buffer.slice(..).get_mapped_range();
        for y in 0..height as usize {
            for x in 0..width as usize {
                let offset = y * padded + x * 16;
                let px = &data[offset..offset + 16];
                pixels.push([
                    f32::from_ne_bytes(px[0..4].try_into().unwrap()),
                    f32::from_ne_bytes(px[4..8].try_into().unwrap()),
                    f32::from_ne_bytes(px[8..12].try_into().unwrap()),
                    f32::from_ne_bytes(px[12..16].try_into().unwrap()),
                ]);
            }
        }
    }
    buffer.unmap();
    pixels
}
pub(super) fn pixel(pixels: &[[f32; 4]], x: u32, y: u32, size: UVec2) -> [f32; 4] {
    pixels[(y * size.x + x) as usize]
}

pub(super) fn wait_stage<T>(
    fixture: &mut Fixture,
    stage: &str,
    mut check: impl FnMut(&mut Fixture) -> Option<T>,
) -> T {
    const MAX_UPDATES: u32 = 240;
    let started = Instant::now();
    let deadline = started + Duration::from_secs(30);
    fixture.stage_deadline = Some(deadline);
    let mut updates = 0;
    loop {
        if updates >= MAX_UPDATES || Instant::now() >= deadline {
            fixture.stage_deadline = None;
            panic!(
                "timed out in {stage}: updates={updates}, elapsed={:?}, {}",
                started.elapsed(),
                fixture.diagnostics()
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

impl Fixture {
    pub(super) fn pipeline_diagnostics(&self) -> String {
        let world = self.render_world();
        let mut lines = Vec::new();
        lines.extend(depth::gpu_pipeline_diagnostics(world));
        lines.extend(motion::gpu_pipeline_diagnostics(world));
        lines.extend(reactive::gpu_pipeline_diagnostics(world));
        lines.extend(rejection::gpu_pipeline_diagnostics(world));
        lines.extend(scene_color::gpu_pipeline_diagnostics(world));
        lines.join("; ")
    }

    fn diagnostics(&self) -> String {
        let Some(slot) = self.slot(self.camera) else {
            return format!(
                "no depth request for camera {:?}; pipelines: {}; warmup={}",
                self.camera,
                self.pipeline_diagnostics(),
                self.warmup_evidence,
            );
        };
        let probes = self.probes(&slot);
        let base = format!(
            "view={:?} key={:?} epoch={} depth={} motion={} reactive={} rejection={} scene={}; pipelines: {}",
            slot.render,
            slot.key,
            slot.epoch,
            present(probes.as_ref().map(|p| p.depth.current.is_some())),
            present(probes.as_ref().map(|p| p.motion.current.is_some())),
            present(probes.as_ref().map(|p| p.reactive.current.is_some())),
            present(probes.as_ref().map(|p| p.rejection.current.is_some())),
            present(probes.as_ref().map(|p| p.scene.current.is_some())),
            self.pipeline_diagnostics(),
        );
        format!("{base}; warmup={}", self.warmup_evidence)
    }
}

fn present(value: Option<bool>) -> &'static str {
    match value {
        Some(true) => "current",
        Some(false) => "allocated-unpublished",
        None => "absent",
    }
}

pub(super) fn wait_probes(fixture: &mut Fixture, main_view: Entity) -> (ViewSlot, Probes) {
    wait_stage(fixture, "temporal outputs", |fixture| {
        let slot = fixture.slot(main_view)?;
        let probes = fixture.probes(&slot)?;
        if probes.depth.current.is_some()
            && probes.depth.previous.is_some()
            && probes.motion.current.is_some()
            && probes.reactive.current.is_some()
            && probes.reactive.previous.is_some()
            && probes.rejection.current.is_some()
            && probes.rejection.previous.is_some()
            && probes.scene.current.is_some()
            && probes.scene.previous.is_some()
        {
            Some((slot, probes))
        } else {
            None
        }
    })
}

pub(super) fn is_approx(a: f32, b: f32, tolerance: f32) -> bool {
    (a - b).abs() <= tolerance
}

#[cfg(test)]
mod tests;
