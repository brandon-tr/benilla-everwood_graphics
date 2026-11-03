//! Deterministic retained-wind GPU fixture and CPU deformation oracle.

use std::sync::Arc;

use bevy::core_pipeline::prepass::NormalPrepass;
use bevy::prelude::*;
use bevy::render::extract_resource::ExtractResourcePlugin;
use bevy::render::renderer::RenderQueue;

use benilla_assets::coords::bevy_to_wow;
use benilla_formats::{ModelBlend, RenderSubmesh};

use crate::interact::WorldObject;
use crate::lighting::{GxRetainedWindFrame, LightBlob, MonkeyFrame, SharedLightBuffer};
use crate::static_gx::{testkit, StaticGx};

use super::RetainedReferenceFixture;

const TAU: f32 = std::f32::consts::TAU;
const WRAP: f32 = 4096.0;

pub(super) struct WindReferenceFixture {
    pub(super) retained: RetainedReferenceFixture,
    pub(super) wind: MonkeyFrame,
    pub(super) anchor: Vec3,
}

const TARGET_WORLD: [Vec3; 4] = [
    Vec3::new(-1.0, 1.0, 0.3),
    Vec3::new(1.0, 1.0, 0.3),
    Vec3::new(1.0, 3.0, 0.3),
    Vec3::new(-1.0, 3.0, 0.3),
];

fn tree_geometry_at_anchor(anchor: Vec3) -> Arc<RenderSubmesh> {
    Arc::new(RenderSubmesh {
        positions: TARGET_WORLD
            .iter()
            .map(|p| bevy_to_wow(*p - anchor))
            .collect(),
        normals: vec![bevy_to_wow(Vec3::Z); 4],
        uvs: vec![[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
        indices: vec![0, 1, 2, 0, 2, 3],
        blend: ModelBlend::AlphaTest,
        wrap_x: true,
        wrap_y: true,
        two_sided: true,
        ..Default::default()
    })
}

fn tree_geometry() -> Arc<RenderSubmesh> {
    tree_geometry_at_anchor(Vec3::new(0.0, 0.0, 0.3))
}

fn tree_object() -> Arc<WorldObject> {
    Arc::new(WorldObject {
        kind: crate::model_render::ModelKind::Doodad,
        label: r"World\Azeroth\Elwynn\PassiveDoodads\Trees\ElwynnTree01.m2".into(),
        id: 7001,
        detail: String::new(),
    })
}

impl WindReferenceFixture {
    pub(super) fn new() -> Self {
        Self::new_with_anchor(Vec3::new(0.0, 0.0, 0.3))
    }
    pub(super) fn new_with_anchor(anchor: Vec3) -> Self {
        let mut retained = RetainedReferenceFixture::new_with_setup(|app| {
            app.add_plugins(ExtractResourcePlugin::<GxRetainedWindFrame>::default());
        });
        retained.hide_terrain(true);
        let camera = retained.terrain.base.camera;
        retained
            .terrain
            .base
            .app
            .world_mut()
            .entity_mut(camera)
            .insert(
                Transform::from_xyz(0.0, 2.0, 5.0).looking_at(Vec3::new(0.0, 2.0, 0.0), Vec3::Y),
            );
        retained
            .terrain
            .base
            .app
            .world_mut()
            .entity_mut(camera)
            .insert(NormalPrepass);
        retained.release_retained();
        let object = tree_object();
        let geometry = tree_geometry_at_anchor(anchor);
        let mut batch = testkit::batch_of(
            &object,
            &geometry,
            anchor,
            Some(retained.cutout.clone()),
            ModelBlend::AlphaTest,
        );
        batch.owner = retained.owner;
        batch.two_sided = true;
        batch.unlit = true;
        let mut gx = retained
            .terrain
            .base
            .app
            .world_mut()
            .resource_mut::<StaticGx>();
        assert!(gx.divert(batch), "tree batch admitted");
        gx.flush_now();
        Self {
            retained,
            wind: MonkeyFrame::default(),
            anchor,
        }
    }

    pub(super) fn set_wind(&mut self, frame: MonkeyFrame) {
        let queue = self
            .retained
            .terrain
            .base
            .render_world()
            .resource::<RenderQueue>()
            .clone();
        let buffer = self
            .retained
            .terrain
            .base
            .app
            .world()
            .resource::<SharedLightBuffer>()
            .0
            .clone();
        let snapshot = LightBlob::write_monkey_frame(&queue, &buffer, &frame, 0.5, 0.0);
        self.retained
            .terrain
            .base
            .app
            .world_mut()
            .insert_resource(snapshot);
        self.wind = frame;
    }

    pub(super) fn set_camera(&mut self, eye: Vec3, target: Vec3) {
        let camera = self.retained.terrain.base.camera;
        self.retained
            .terrain
            .base
            .app
            .world_mut()
            .entity_mut(camera)
            .insert(Transform::from_translation(eye).looking_at(target, Vec3::Y));
    }

    pub(super) fn update(&mut self) {
        self.retained.update();
    }
}

fn wind_base(frame: &MonkeyFrame) -> Vec2 {
    Vec2::new(frame.wind_base_heading.cos(), frame.wind_base_heading.sin())
}

fn wgsl_fract(x: f32) -> f32 {
    x - x.floor()
}

fn phase(travel: f32, cycles: f32) -> f32 {
    wgsl_fract(cycles * (travel / WRAP)) * TAU
}

fn gust_roll(frame: &MonkeyFrame, xz: Vec2) -> f32 {
    let band = (wind_base(frame).dot(xz) * (TAU / 55.0) - phase(frame.wind_travel, 68.0)).sin();
    0.55 + 0.65 * (0.5 + 0.5 * band)
}

fn smoothstep(a: f32, b: f32, x: f32) -> f32 {
    let t = ((x - a) / (b - a)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// CPU mirror of `wind_hook::tree_offset`; test oracle only.
pub(super) fn tree_offset_reference(
    world: Vec3,
    anchor: Vec3,
    camera: Vec3,
    frame: &MonkeyFrame,
) -> Vec3 {
    if frame.tree_strength <= 0.0 {
        return Vec3::ZERO;
    }
    let h = (world.y - anchor.y).max(0.0);
    let crown = smoothstep(0.15, 1.5, h);
    let weight = crown * crown * (0.25 + h * 0.12).clamp(0.25, 1.3);
    let dir =
        (Vec2::new(frame.wind_dir[0], frame.wind_dir[1]) + Vec2::new(1.0e-6, 0.0)).normalize();
    let cross = Vec2::new(-dir.y, dir.x);
    let base = wind_base(frame);
    let base_cross = Vec2::new(-base.y, base.x);
    let p = wgsl_fract(
        (anchor.xz().dot(Vec2::new(0.173, 0.317)).sin() * 43_758.547).fract() * TAU / TAU,
    ) * TAU;
    let slow = (base.dot(world.xz()) * (TAU / 34.0) - phase(frame.wind_travel, 114.0) + p).sin();
    let side = (base_cross.dot(world.xz()) * (TAU / 21.0) - phase(frame.wind_travel, 82.0)
        + p * 1.7)
        .sin();
    let flutter =
        (phase(frame.wind_travel, 569.0) + world.xz().dot(Vec2::new(1.7, 2.1)) + p * 2.0).sin();
    let gust = 0.75 + frame.wind_gust * 0.5;
    let fade = 1.0 / (1.0 + camera.distance(world) * 0.006);
    let xz = (dir * (0.08 + slow * 0.20) + cross * side * 0.08 + dir * flutter * 0.03)
        * weight
        * gust
        * (0.75 + 0.30 * ((gust_roll(frame, anchor.xz()) - 0.55) / 0.65))
        * fade
        * frame.sway_strength
        * frame.tree_strength;
    Vec3::new(xz.x, -0.15 * xz.dot(xz), xz.y)
}

#[cfg(test)]
mod camera;
#[cfg(test)]
mod tests;

#[allow(dead_code)]
pub(super) struct OracleSample {
    pub weights: [f32; 3],
    pub current_world: Vec3,
    pub previous_world: Vec3,
    pub motion: Vec2,
}

fn project_ndc(m: Mat4, p: Vec3) -> Option<(Vec2, f32)> {
    let c = m * p.extend(1.0);
    if c.w <= 0.0 || !c.is_finite() {
        return None;
    }
    Some((c.xy() / c.w, c.w))
}

pub(super) fn expected_triangle_motion(
    pixel_center: Vec2,
    viewport: UVec2,
    current_world: [Vec3; 3],
    previous_world: [Vec3; 3],
    current_raster_clip_from_world: Mat4,
    current_unjittered_clip_from_world: Mat4,
    previous_unjittered_clip_from_world: Mat4,
) -> Option<OracleSample> {
    if viewport.x == 0 || viewport.y == 0 {
        return None;
    }
    let mut screen = [Vec2::ZERO; 3];
    let mut cw = [0.0; 3];
    for i in 0..3 {
        let (ndc, w) = project_ndc(current_raster_clip_from_world, current_world[i])?;
        screen[i] = Vec2::new(
            (ndc.x * 0.5 + 0.5) * viewport.x as f32,
            (0.5 - ndc.y * 0.5) * viewport.y as f32,
        );
        cw[i] = w;
    }
    let den = (screen[1].y - screen[2].y) * (screen[0].x - screen[2].x)
        + (screen[2].x - screen[1].x) * (screen[0].y - screen[2].y);
    if !den.is_finite() || den.abs() < 1.0e-8 {
        return None;
    }
    let l0 = ((screen[1].y - screen[2].y) * (pixel_center.x - screen[2].x)
        + (screen[2].x - screen[1].x) * (pixel_center.y - screen[2].y))
        / den;
    let l1 = ((screen[2].y - screen[0].y) * (pixel_center.x - screen[2].x)
        + (screen[0].x - screen[2].x) * (pixel_center.y - screen[2].y))
        / den;
    let l = [l0, l1, 1.0 - l0 - l1];
    if l.iter().any(|x| !x.is_finite() || *x < 0.0 || *x > 1.0) {
        return None;
    }
    let q = [l[0] / cw[0], l[1] / cw[1], l[2] / cw[2]];
    let sum = q[0] + q[1] + q[2];
    if !sum.is_finite() || sum.abs() < 1.0e-8 {
        return None;
    }
    let w = [q[0] / sum, q[1] / sum, q[2] / sum];
    let cur = current_world[0] * w[0] + current_world[1] * w[1] + current_world[2] * w[2];
    let prev = previous_world[0] * w[0] + previous_world[1] * w[1] + previous_world[2] * w[2];
    let (cur_ndc, _) = project_ndc(current_unjittered_clip_from_world, cur)?;
    let (prev_ndc, _) = project_ndc(previous_unjittered_clip_from_world, prev)?;
    let motion = (cur_ndc - prev_ndc) * Vec2::new(0.5, -0.5);
    if !motion.is_finite() {
        return None;
    }
    Some(OracleSample {
        weights: w,
        current_world: cur,
        previous_world: prev,
        motion,
    })
}

#[cfg(test)]
mod gpu_reference;
