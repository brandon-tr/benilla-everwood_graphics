//! Test-only stock-TAA fixture for real retained static_gx geometry.

use std::sync::Arc;

use bevy::asset::RenderAssetUsages;
use bevy::image::Image;
use bevy::prelude::*;
use bevy::render::extract_resource::ExtractResourcePlugin;
use bevy::render::render_resource::{Extent3d, TextureDimension, TextureFormat};
use bevy::render::renderer::{RenderDevice, RenderQueue};

use benilla_assets::coords::bevy_to_wow;
use benilla_formats::{ModelBlend, RenderSubmesh};

use crate::dev_state::DebugState;
use crate::lighting::{LightBlob, SharedLightBuffer};
use crate::static_gx::{new_torch_shared, register_torch_shared, StaticGx, StaticGxPlugin};
use crate::view::{ViewDistance, WorldCamera};
use crate::wmo_portal::{CameraInteriorClaim, ExteriorWindows};

use super::TerrainReferenceFixture;

pub(super) struct RetainedReferenceFixture {
    pub(super) terrain: TerrainReferenceFixture,
    pub(super) owner: (i32, i32),
    pub(super) opaque: Handle<Image>,
    pub(super) cutout: Handle<Image>,
}

fn retained_submesh(blend: ModelBlend) -> Arc<RenderSubmesh> {
    // The intended surface is a Bevy-space XY card at local Z=0, normal +Z. Bake converts WoW
    // coordinates with (-y, z, -x), so store the inverse-converted source data.
    let bevy_positions = [
        Vec3::new(-1.5, -1.5, 0.0),
        Vec3::new(1.5, -1.5, 0.0),
        Vec3::new(1.5, 1.5, 0.0),
        Vec3::new(-1.5, 1.5, 0.0),
    ];
    let positions = bevy_positions
        .iter()
        .map(|p| bevy_to_wow(*p))
        .collect::<Vec<_>>();
    let normal = bevy_to_wow(Vec3::Z);
    Arc::new(RenderSubmesh {
        positions,
        normals: vec![normal; 4],
        uvs: vec![[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
        indices: vec![0, 1, 2, 0, 2, 3],
        blend,
        wrap_x: true,
        wrap_y: true,
        two_sided: true,
        ..Default::default()
    })
}

fn image(data: Vec<u8>) -> Image {
    Image::new(
        Extent3d {
            width: 8,
            height: 8,
            depth_or_array_layers: 1,
        },
        TextureDimension::D2,
        data,
        TextureFormat::Rgba8Unorm,
        RenderAssetUsages::RENDER_WORLD,
    )
}

fn opaque_texture() -> Image {
    image([255u8, 96, 32, 255].repeat(8 * 8))
}

fn cutout_texture() -> Image {
    let mut data = [255u8, 0, 255, 255].repeat(8 * 8);
    // A real transparent hole at the sampled centre; the card is opaque everywhere else.
    for y in 3..5 {
        for x in 3..5 {
            let i = (y * 8 + x) * 4;
            data[i + 3] = 0;
        }
    }
    image(data)
}

impl RetainedReferenceFixture {
    pub(super) fn new() -> Self {
        Self::new_with_setup(|_| {})
    }

    pub(super) fn new_with_setup(setup: impl FnOnce(&mut App)) -> Self {
        let mut terrain = TerrainReferenceFixture::new_with_setup(|app| {
            crate::shaders::plugin(app);
            app.add_plugins(StaticGxPlugin);
            app.add_plugins(ExtractResourcePlugin::<SharedLightBuffer>::default());
            register_torch_shared(app);
            setup(app);
        });

        let camera = terrain.base.camera;
        terrain
            .base
            .app
            .world_mut()
            .entity_mut(camera)
            .insert(WorldCamera);

        let world = terrain.base.app.world_mut();
        world.init_resource::<DebugState>();
        world.init_resource::<ViewDistance>();
        world.init_resource::<ExteriorWindows>();
        world.init_resource::<CameraInteriorClaim>();

        let device = terrain
            .base
            .render_world()
            .resource::<RenderDevice>()
            .clone();
        let queue = terrain
            .base
            .render_world()
            .resource::<RenderQueue>()
            .clone();

        let light = LightBlob::model([1.0; 3], [0.0; 3], Vec3::NEG_Z);
        let light_buf = light.create(&device, "retained native taa light");
        light.write(&queue, &light_buf);
        terrain
            .base
            .app
            .world_mut()
            .insert_resource(SharedLightBuffer(light_buf));

        let (torch_image, torch_buffer) = {
            let mut images = terrain.base.app.world_mut().resource_mut::<Assets<Image>>();
            new_torch_shared(&device, &mut images)
        };
        terrain.base.app.world_mut().insert_resource(torch_image);
        terrain.base.app.world_mut().insert_resource(torch_buffer);

        let mut images = terrain.base.app.world_mut().resource_mut::<Assets<Image>>();
        let opaque = images.add(opaque_texture());
        let cutout = images.add(cutout_texture());

        let mut fixture = Self {
            terrain,
            owner: (0, 0),
            opaque,
            cutout,
        };
        fixture.set_opaque(Vec3::new(0.0, 0.0, 0.05), true, 0.0);
        fixture
    }

    fn replace(
        &mut self,
        texture: Handle<Image>,
        blend: ModelBlend,
        transform: Transform,
        two_sided: bool,
    ) {
        self.release_retained();
        let object = crate::static_gx::testkit::object(1);
        let geometry = retained_submesh(blend);
        let mut batch = crate::static_gx::testkit::batch_of(
            &object,
            &geometry,
            transform.translation,
            Some(texture),
            blend,
        );
        batch.transform = transform;
        batch.owner = self.owner;
        batch.two_sided = two_sided;
        batch.unlit = true;
        let mut gx = self.terrain.base.app.world_mut().resource_mut::<StaticGx>();
        assert!(gx.divert(batch), "real retained batch must be admitted");
        gx.flush_now();
    }

    pub(super) fn set_opaque(&mut self, at: Vec3, two_sided: bool, yaw: f32) {
        let texture = self.opaque.clone();
        self.replace(
            texture,
            ModelBlend::Opaque,
            Transform::from_translation(at).with_rotation(Quat::from_rotation_y(yaw)),
            two_sided,
        );
    }

    pub(super) fn set_cutout(&mut self, at: Vec3) {
        let texture = self.cutout.clone();
        self.replace(
            texture,
            ModelBlend::AlphaTest,
            Transform::from_translation(at),
            true,
        );
    }

    pub(super) fn set_cutout_yaw(&mut self, at: Vec3, yaw: f32) {
        let texture = self.cutout.clone();
        self.replace(
            texture,
            ModelBlend::AlphaTest,
            Transform::from_translation(at).with_rotation(Quat::from_rotation_y(yaw)),
            true,
        );
    }

    pub(super) fn set_terrain_depth(&mut self, z: f32) {
        let mut entity = self
            .terrain
            .base
            .app
            .world_mut()
            .entity_mut(self.terrain.terrain);
        let mut transform = entity.get_mut::<Transform>().expect("terrain transform");
        transform.translation.z = z;
    }

    pub(super) fn update(&mut self) {
        self.terrain.update();
    }

    pub(super) fn pair(&self) -> (super::ObservedCapture, super::ObservedCapture) {
        self.terrain.pair()
    }

    pub(super) fn read_depth(&self, capture: &super::ObservedCapture) -> Vec<[f32; 4]> {
        self.terrain.read_depth(capture)
    }

    pub(super) fn hide_terrain(&mut self, hidden: bool) {
        self.terrain
            .base
            .app
            .world_mut()
            .entity_mut(self.terrain.terrain)
            .insert(if hidden {
                Visibility::Hidden
            } else {
                Visibility::Visible
            });
    }

    pub(super) fn release_retained(&mut self) {
        self.terrain
            .base
            .app
            .world_mut()
            .resource_mut::<StaticGx>()
            .release_owner(self.owner);
    }

    pub(super) fn clear_retained(&mut self) {
        self.terrain
            .base
            .app
            .world_mut()
            .resource_mut::<StaticGx>()
            .clear();
    }

    pub(super) fn wait<T>(
        &mut self,
        stage: &str,
        check: impl FnMut(&mut TerrainReferenceFixture) -> Option<T>,
    ) -> T {
        self.terrain.wait(stage, check)
    }
}

#[cfg(test)]
mod tests;
#[cfg(test)]
mod wind;
