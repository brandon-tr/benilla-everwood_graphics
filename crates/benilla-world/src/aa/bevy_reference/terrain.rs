//! Test-only stock-TAA fixture for the real Everwood `TerrainMaterial` and `chunk_to_mesh` path.

use std::time::{Duration, Instant};

use bevy::asset::RenderAssetUsages;
use bevy::camera::primitives::MeshAabb;
use bevy::core_pipeline::prepass::ViewPrepassTextures;
use bevy::image::Image;
use bevy::pbr::{MaterialPlugin, MeshMaterial3d};
use bevy::prelude::*;
use bevy::render::render_resource::{
    BindGroupLayoutDescriptor, BindGroupLayoutEntry, BindingType, CachedRenderPipelineId,
    ColorTargetState, ColorWrites, Extent3d, FragmentState, MultisampleState, PipelineCache,
    PrimitiveState, PrimitiveTopology, RenderPipelineDescriptor, ShaderStages, Texture,
    TextureDimension, TextureFormat, TextureSampleType, TextureViewDescriptor,
    TextureViewDimension, VertexState,
};
use bevy::render::renderer::{RenderDevice, RenderQueue};
use bevy::render::RenderApp;
use bevy::shader::Shader;

use benilla_assets::coords::bevy_to_wow;
use benilla_assets::materials::{register_shaders, TerrainExtension, TerrainMaterial, TorchBinds};
use benilla_assets::{chunk_to_mesh, ChunkShading};
use benilla_formats::ChunkMesh;
use benilla_formats::TERRAIN_LAYER_TILES;

use crate::lighting::LightBlob;
use crate::static_gx::new_torch_shared;

use super::{read_texture_pixels, BevyReferenceFixture, ObservedCapture};

const DEPTH_SAMPLE_WGSL: &str = r#"
@group(0) @binding(0) var source: texture_depth_2d;

@vertex
fn vertex(@builtin(vertex_index) i: u32) -> @builtin(position) vec4<f32> {
    let x = f32((i << 1u) & 2u);
    let y = f32(i & 2u);
    return vec4<f32>(x * 2.0 - 1.0, y * 2.0 - 1.0, 0.0, 1.0);
}

@fragment
fn sample(@builtin(position) p: vec4<f32>) -> @location(0) vec4<f32> {
    let depth = textureLoad(source, vec2<i32>(p.xy), 0);
    return vec4<f32>(depth, depth, depth, 1.0);
}
"#;

pub(super) struct TerrainReferenceFixture {
    pub(super) base: BevyReferenceFixture,
    pub(super) terrain: Entity,
    pub(super) last_pixels: String,
    depth_layout: BindGroupLayoutDescriptor,
    depth_pipeline: CachedRenderPipelineId,
}

fn array_image(
    width: u32,
    height: u32,
    layers: u32,
    format: TextureFormat,
    data: Vec<u8>,
) -> Image {
    let mut image = Image::new(
        Extent3d {
            width,
            height,
            depth_or_array_layers: layers,
        },
        TextureDimension::D2,
        data,
        format,
        RenderAssetUsages::RENDER_WORLD,
    );
    image.texture_view_descriptor = Some(TextureViewDescriptor {
        dimension: Some(TextureViewDimension::D2Array),
        ..default()
    });
    image
}

pub(super) fn terrain_mesh() -> Option<Mesh> {
    let bevy_positions = [
        Vec3::new(-1.5, -1.5, 0.0),
        Vec3::new(1.5, -1.5, 0.0),
        Vec3::new(1.5, 1.5, 0.0),
        Vec3::new(-1.5, 1.5, 0.0),
    ];
    let bevy_normals = [Vec3::Z; 4];
    let chunk = ChunkMesh {
        positions: bevy_positions.iter().map(|p| bevy_to_wow(*p)).collect(),
        normals: bevy_normals.iter().map(|n| bevy_to_wow(*n)).collect(),
        uvs: vec![[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
        indices: vec![0, 1, 2, 0, 2, 3],
        holes: 0,
        base_texture: None,
        layer_textures: Vec::new(),
        layer_effect_ids: Vec::new(),
        alpha_map: None,
        shadow: None,
        pred_tex: [0; 64],
        no_effect_doodad: [false; 64],
        index_x: 0,
        index_y: 0,
        area_id: 0,
        impassable: false,
        liquids: Vec::new(),
    };
    chunk_to_mesh(&chunk, &ChunkShading::default())
}

impl TerrainReferenceFixture {
    pub(super) fn new() -> Self {
        Self::new_with_setup(|_| {})
    }

    pub(super) fn new_with_setup(setup: impl FnOnce(&mut App)) -> Self {
        let mut base = BevyReferenceFixture::new_with_setup(|app| {
            register_shaders(app);
            app.add_plugins(MaterialPlugin::<TerrainMaterial>::default());
            setup(app);
        });

        let standard = {
            let mut query = base
                .app
                .world_mut()
                .query_filtered::<Entity, With<MeshMaterial3d<StandardMaterial>>>();
            query.iter(base.app.world()).collect::<Vec<_>>()
        };
        assert_eq!(
            standard.len(),
            1,
            "the isolated reference fixture must own exactly one StandardMaterial scene mesh"
        );
        base.app.world_mut().entity_mut(standard[0]).despawn();

        let device = base.render_world().resource::<RenderDevice>().clone();
        let queue = base.render_world().resource::<RenderQueue>().clone();

        let (torch_image, torch_buffer) = {
            let mut images = base.app.world_mut().resource_mut::<Assets<Image>>();
            new_torch_shared(&device, &mut images)
        };
        let torch = TorchBinds {
            depth: torch_image.0.clone(),
            table: torch_buffer.0.clone(),
        };

        let layer_data = [51u8, 153, 230, 0]
            .iter()
            .copied()
            .cycle()
            .take(64 * 64 * 4 * 4)
            .collect();
        let layer_array = base
            .app
            .world_mut()
            .resource_mut::<Assets<Image>>()
            .add(array_image(
                64,
                64,
                4,
                TextureFormat::Rgba8Unorm,
                layer_data,
            ));
        let alpha_array = base
            .app
            .world_mut()
            .resource_mut::<Assets<Image>>()
            .add(array_image(
                64,
                64,
                1,
                TextureFormat::Rgba8Unorm,
                vec![0; 64 * 64 * 4],
            ));
        let shadow_array = base
            .app
            .world_mut()
            .resource_mut::<Assets<Image>>()
            .add(array_image(
                64,
                64,
                1,
                TextureFormat::R8Unorm,
                vec![0; 64 * 64],
            ));

        let light = LightBlob::model([1.0; 3], [0.0; 3], Vec3::NEG_Z);
        let light_buf = light.create(&device, "terrain native taa light");
        light.write(&queue, &light_buf);

        let material = base
            .app
            .world_mut()
            .resource_mut::<Assets<TerrainMaterial>>()
            .add(TerrainMaterial {
                base: StandardMaterial {
                    base_color: Color::WHITE,
                    perceptual_roughness: 1.0,
                    double_sided: false,
                    cull_mode: Some(bevy::render::render_resource::Face::Back),
                    ..default()
                },
                extension: TerrainExtension {
                    layer_array,
                    alpha_array,
                    shadow_array,
                    params: Vec4::new(TERRAIN_LAYER_TILES, 0.0, 0.0, 0.0),
                    light_buf,
                    torch_depth: torch.depth,
                    torch_buf: torch.table,
                },
            });

        let mesh = terrain_mesh().expect("synthetic terrain mesh");
        let aabb = mesh.compute_aabb();
        let mesh = base
            .app
            .world_mut()
            .resource_mut::<Assets<Mesh>>()
            .add(mesh);
        let mut terrain_entity = base.app.world_mut().spawn((
            Mesh3d(mesh),
            MeshMaterial3d(material),
            Transform::from_rotation(Quat::from_rotation_z(0.23)),
        ));
        if let Some(aabb) = aabb {
            terrain_entity.insert(aabb);
        }
        let terrain = terrain_entity.id();

        let shader = base
            .app
            .world_mut()
            .resource_mut::<Assets<Shader>>()
            .add(Shader::from_wgsl(
                DEPTH_SAMPLE_WGSL,
                "terrain_depth_sample.wgsl",
            ));
        let depth_layout = BindGroupLayoutDescriptor::new(
            "terrain depth sampling",
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
        let depth_pipeline = base
            .app
            .get_sub_app_mut(RenderApp)
            .expect("render app")
            .world_mut()
            .resource_mut::<PipelineCache>()
            .queue_render_pipeline(RenderPipelineDescriptor {
                label: Some("terrain depth sampling".into()),
                layout: vec![depth_layout.clone()],
                vertex: VertexState {
                    shader: shader.clone(),
                    entry_point: Some("vertex".into()),
                    ..default()
                },
                fragment: Some(FragmentState {
                    shader,
                    entry_point: Some("sample".into()),
                    targets: vec![Some(ColorTargetState {
                        format: TextureFormat::Rgba32Float,
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

        Self {
            base,
            terrain,
            last_pixels: String::new(),
            depth_layout,
            depth_pipeline,
        }
    }

    pub(super) fn update(&mut self) {
        self.base.update();
    }

    pub(super) fn pair(&self) -> (ObservedCapture, ObservedCapture) {
        let (before, after) = self.base.captures();
        let before = before.expect("before-TAA capture");
        let after = after.expect("after-TAA capture");
        assert_eq!(
            before.view, after.view,
            "paired observations must be the same view"
        );
        assert_eq!(
            before.frame, after.frame,
            "paired observations must be the same frame"
        );
        (before, after)
    }

    pub(super) fn sampling_ready(&self) -> bool {
        self.base.sampling_ready()
            && self
                .base
                .render_world()
                .resource::<PipelineCache>()
                .get_render_pipeline(self.depth_pipeline)
                .is_some()
    }

    pub(super) fn depth_texture(&self, capture: &ObservedCapture) -> Option<Texture> {
        self.base
            .render_world()
            .get::<ViewPrepassTextures>(capture.view)?
            .depth
            .as_ref()
            .map(|attachment| attachment.texture.texture.clone())
    }

    pub(super) fn read_depth(&self, capture: &ObservedCapture) -> Vec<[f32; 4]> {
        let depth = self.depth_texture(capture).expect("native prepass depth");
        read_texture_pixels(
            self.base.render_world(),
            &self.depth_layout,
            self.depth_pipeline,
            self.base.stage_deadline,
            &depth,
        )
    }

    pub(super) fn set_camera_translation(&mut self, x: f32) {
        let mut entity = self.base.app.world_mut().entity_mut(self.base.camera);
        entity.insert(Transform::from_xyz(x, 0.0, 5.0).looking_at(Vec3::new(x, 0.0, 0.0), Vec3::Y));
    }

    pub(super) fn wait<T>(
        &mut self,
        stage: &str,
        mut check: impl FnMut(&mut Self) -> Option<T>,
    ) -> T {
        const MAX_UPDATES: u32 = 240;
        let started = Instant::now();
        let deadline = started + Duration::from_secs(30);
        self.base.stage_deadline = Some(deadline);
        self.base.stage_evidence.clear();
        let mut updates = 0;
        loop {
            if updates >= MAX_UPDATES || Instant::now() >= deadline {
                self.base.stage_deadline = None;
                let captures = self.base.captures();
                panic!(
                    "timed out in {stage}: updates={updates}, elapsed={:?}, sampling={}, before={:?}, after={:?}, evidence={}",
                    started.elapsed(),
                    self.sampling_ready(),
                    captures
                        .0
                        .map(|capture| (capture.frame, capture.reset, capture.taa_present)),
                    captures
                        .1
                        .map(|capture| (capture.frame, capture.reset, capture.taa_present)),
                    self.base.stage_evidence
                );
            }
            self.base.update();
            updates += 1;
            let captures = self.base.captures();
            self.base.stage_evidence = format!(
                "updates={updates} sampling={} before={:?} after={:?} pixels={}",
                self.sampling_ready(),
                captures.0.as_ref().map(|capture| (
                    capture.frame,
                    capture.view,
                    capture.reset,
                    capture.taa_present
                )),
                captures.1.as_ref().map(|capture| (
                    capture.frame,
                    capture.view,
                    capture.reset,
                    capture.taa_present
                )),
                self.last_pixels
            );
            if let Some(value) = check(self) {
                self.base.stage_deadline = None;
                return value;
            }
        }
    }

    pub(super) fn print_adapter(&self) {
        self.base.print_adapter();
        let info = self
            .base
            .app
            .world()
            .resource::<bevy::render::renderer::RenderAdapterInfo>();
        println!(
            "terrain native TAA adapter backend={:?} device_type={:?}",
            info.backend, info.device_type
        );
    }
}

pub(super) fn pixel_index(x: u32, y: u32) -> usize {
    (y * super::SIZE.x + x) as usize
}

#[cfg(test)]
mod retained;
#[cfg(test)]
mod tests;
