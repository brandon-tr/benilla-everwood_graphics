//! Test-only vertex evaluation of the production wind hook.

use bevy::prelude::*;
use bevy::render::render_resource::*;
use bevy::render::renderer::{RenderDevice, RenderQueue};
use bevy::render::RenderApp;
use bevy::shader::Shader;
use std::num::NonZeroU64;

use super::{MonkeyFrame, WindReferenceFixture, TARGET_WORLD};

const WGSL: &str = r#"
#import benilla::wind_hook
#import benilla::monkey_frame

struct Inputs {
    positions: array<vec4<f32>, 4>,
    anchor: vec4<f32>,
    camera: vec4<f32>,
    wind_a: vec4<f32>,
    wind_b: vec4<f32>,
};

@group(0) @binding(0) var<uniform> inputs: Inputs;

struct Out {
    @builtin(position) position: vec4<f32>,
    @location(0) @interpolate(flat) world: vec4<f32>,
};

@vertex
fn vertex(@builtin(vertex_index) index: u32) -> Out {
    let base = inputs.positions[index].xyz;
    var frame: monkey_frame::MonkeyFrame;
    frame.wind_a = inputs.wind_a;
    frame.wind_b = inputs.wind_b;
    let world = base + wind_hook::tree_offset(base, inputs.anchor.xyz, inputs.camera.xyz, frame);
    var out: Out;
    let x = 2.0 * ((f32(index) + 0.5) / 4.0) - 1.0;
    out.position = vec4<f32>(x, 0.0, 0.0, 1.0);
    out.world = vec4<f32>(world, 1.0);
    return out;
}

@fragment
fn fragment(in: Out) -> @location(0) vec4<f32> {
    return in.world;
}
"#;

pub(super) const INPUT_BYTES: u64 = std::mem::size_of::<[[f32; 4]; 8]>() as u64;

pub(super) fn reference_input_rows(
    anchor: Vec3,
    camera: Vec3,
    frame: &MonkeyFrame,
) -> [[f32; 4]; 8] {
    let packed = frame.pack(0.5, 0.0);
    [
        [TARGET_WORLD[0].x, TARGET_WORLD[0].y, TARGET_WORLD[0].z, 1.0],
        [TARGET_WORLD[1].x, TARGET_WORLD[1].y, TARGET_WORLD[1].z, 1.0],
        [TARGET_WORLD[2].x, TARGET_WORLD[2].y, TARGET_WORLD[2].z, 1.0],
        [TARGET_WORLD[3].x, TARGET_WORLD[3].y, TARGET_WORLD[3].z, 1.0],
        [anchor.x, anchor.y, anchor.z, 1.0],
        [camera.x, camera.y, camera.z, 1.0],
        packed[4],
        packed[5],
    ]
}

pub(super) struct WindGpuReference {
    pipeline: CachedRenderPipelineId,
    buffer: Buffer,
    bind: BindGroup,
    target: Texture,
}

impl WindGpuReference {
    pub(super) fn new(app: &mut App) -> Self {
        let shader = app
            .world_mut()
            .resource_mut::<Assets<Shader>>()
            .add(Shader::from_wgsl(WGSL, "wind_gpu_reference.wgsl"));
        let layout = BindGroupLayoutDescriptor::new(
            "wind_gpu_reference_layout",
            &[BindGroupLayoutEntry {
                binding: 0,
                visibility: ShaderStages::VERTEX,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: NonZeroU64::new(INPUT_BYTES),
                },
                count: None,
            }],
        );
        let render_app = app.get_sub_app_mut(RenderApp).expect("render app");
        let world = render_app.world_mut();
        let pipeline =
            world
                .resource_mut::<PipelineCache>()
                .queue_render_pipeline(RenderPipelineDescriptor {
                    label: Some("wind_gpu_reference".into()),
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
                            format: TextureFormat::Rgba32Float,
                            blend: None,
                            write_mask: ColorWrites::ALL,
                        })],
                        ..default()
                    }),
                    primitive: PrimitiveState {
                        topology: PrimitiveTopology::PointList,
                        ..default()
                    },
                    multisample: MultisampleState {
                        count: 1,
                        ..default()
                    },
                    ..default()
                });
        let device = world.resource::<RenderDevice>().clone();
        let buffer = device.create_buffer(&BufferDescriptor {
            label: Some("wind_gpu_reference_uniform"),
            size: INPUT_BYTES,
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let bind = device.create_bind_group(
            "wind_gpu_reference_bind",
            &world
                .resource::<PipelineCache>()
                .get_bind_group_layout(&layout),
            &BindGroupEntries::single(buffer.as_entire_binding()),
        );
        let target = device.create_texture(&TextureDescriptor {
            label: Some("wind_gpu_reference_target"),
            size: Extent3d {
                width: 4,
                height: 1,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: TextureFormat::Rgba32Float,
            usage: TextureUsages::RENDER_ATTACHMENT | TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        Self {
            pipeline,
            buffer,
            bind,
            target,
        }
    }

    pub(super) fn is_ready(&self, world: &World) -> bool {
        world
            .resource::<PipelineCache>()
            .get_render_pipeline(self.pipeline)
            .is_some()
    }

    pub(super) fn evaluate(
        &self,
        fixture: &WindReferenceFixture,
        frame: &MonkeyFrame,
        camera: Vec3,
    ) -> [Vec3; 4] {
        let base = &fixture.retained.terrain.base;
        let world = base.render_world();
        let device = world.resource::<RenderDevice>();
        let queue = world.resource::<RenderQueue>();
        let pipeline = world
            .resource::<PipelineCache>()
            .get_render_pipeline(self.pipeline)
            .expect("wind gpu reference pipeline");
        let rows = reference_input_rows(fixture.anchor, camera, frame);
        let bytes = bytemuck::cast_slice(&rows);
        assert_eq!(bytes.len() as u64, INPUT_BYTES, "reference uniform bytes");
        queue.write_buffer(&self.buffer, 0, bytes);
        let view = self.target.create_view(&TextureViewDescriptor::default());
        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("wind_gpu_reference"),
        });
        {
            let mut pass = encoder.begin_render_pass(&RenderPassDescriptor {
                label: Some("wind_gpu_reference"),
                color_attachments: &[Some(RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: Operations {
                        load: LoadOp::Clear(Default::default()),
                        store: StoreOp::Store,
                    },
                    depth_slice: None,
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, &self.bind, &[]);
            pass.draw(0..4, 0..1);
        }
        queue.submit(Some(encoder.finish()));
        let pixels = base.read_texture(&self.target);
        assert_eq!(pixels.len(), 4, "four reference vertices");
        let mut out = [Vec3::ZERO; 4];
        for (i, p) in pixels.iter().enumerate() {
            assert!((p[3] - 1.0).abs() < 1.0e-3, "alpha {i}");
            assert!(
                p[0].is_finite() && p[1].is_finite() && p[2].is_finite(),
                "world {i}"
            );
            out[i] = Vec3::new(p[0], p[1], p[2]);
        }
        out
    }
}
