//! Test-only moving-camera/fade validation for retained wind.

use bevy::core_pipeline::prepass::ViewPrepassTextures;
use bevy::prelude::*;
use bevy::render::camera::TemporalJitter;
use bevy::render::view::ExtractedView;

use super::{
    expected_triangle_motion, tree_offset_reference, MonkeyFrame, WindReferenceFixture,
    TARGET_WORLD,
};

pub(super) struct CameraMotionSample {
    pub(super) pixel: Vec2,
    pub(super) index: usize,
    pub(super) triangle: usize,
    pub(super) expected: Vec2,
    pub(super) wrong_previous: Vec2,
}

pub(super) fn camera_motion_samples(
    current: [Vec3; 4],
    previous: [Vec3; 4],
    wrong_previous: [Vec3; 4],
    current_raster: Mat4,
    current_unjittered: Mat4,
    previous_unjittered: Mat4,
) -> Vec<CameraMotionSample> {
    let viewport = UVec2::splat(64);
    let uvs = [Vec2::ZERO, Vec2::X, Vec2::ONE, Vec2::Y];
    let mut seen = [false; 4096];
    let mut out = Vec::new();
    for y in 0..64u32 {
        for x in 0..64u32 {
            let pixel = Vec2::new(x as f32 + 0.5, y as f32 + 0.5);
            let index = (y * 64 + x) as usize;
            if seen[index] {
                continue;
            }
            for (ti, tri) in [[0usize, 1, 2], [0, 2, 3]].into_iter().enumerate() {
                let cw = [current[tri[0]], current[tri[1]], current[tri[2]]];
                let pw = [previous[tri[0]], previous[tri[1]], previous[tri[2]]];
                let Some(s) = expected_triangle_motion(
                    pixel,
                    viewport,
                    cw,
                    pw,
                    current_raster,
                    current_unjittered,
                    previous_unjittered,
                ) else {
                    continue;
                };
                if s.weights.iter().any(|w| *w <= 0.08) {
                    continue;
                }
                let uv = uvs[tri[0]] * s.weights[0]
                    + uvs[tri[1]] * s.weights[1]
                    + uvs[tri[2]] * s.weights[2];
                if uv.x < 0.1 || uv.x > 0.9 || uv.y < 0.1 || uv.y > 0.9 {
                    continue;
                }
                if !(uv.x < 0.22 || uv.x > 0.78 || uv.y < 0.22 || uv.y > 0.78) {
                    continue;
                }
                let ww = [
                    wrong_previous[tri[0]],
                    wrong_previous[tri[1]],
                    wrong_previous[tri[2]],
                ];
                let Some(w) = expected_triangle_motion(
                    pixel,
                    viewport,
                    cw,
                    ww,
                    current_raster,
                    current_unjittered,
                    previous_unjittered,
                ) else {
                    continue;
                };
                seen[index] = true;
                out.push(CameraMotionSample {
                    pixel,
                    index,
                    triangle: ti,
                    expected: s.motion,
                    wrong_previous: w.motion,
                });
                break;
            }
        }
    }
    out
}

fn look(eye: Vec3, target: Vec3) -> Mat4 {
    Mat4::look_at_rh(eye, target, Vec3::Y)
}

fn frame() -> MonkeyFrame {
    MonkeyFrame {
        wind_dir: [0.6, 0.8],
        wind_base_heading: 0.7,
        wind_gust: 0.5,
        wind_travel: 150.0,
        sway_strength: 32.0,
        tree_strength: 1.0,
        ..Default::default()
    }
}

fn vertices(camera: Vec3, wind: &MonkeyFrame) -> [Vec3; 4] {
    let mut out = [Vec3::ZERO; 4];
    for i in 0..4 {
        out[i] = TARGET_WORLD[i] + tree_offset_reference(TARGET_WORLD[i], Vec3::ZERO, camera, wind);
    }
    out
}

pub(super) struct CameraFrame {
    pub(super) view: Entity,
    pub(super) sequence: u64,
    pub(super) unjittered: Mat4,
    pub(super) raster: Mat4,
    pub(super) camera: Vec3,
    pub(super) before: crate::aa::bevy_reference::ObservedCapture,
    pub(super) after: crate::aa::bevy_reference::ObservedCapture,
}

pub(super) fn observed_frame(
    terrain: &crate::aa::bevy_reference::terrain::TerrainReferenceFixture,
    expected_reset: bool,
) -> Option<CameraFrame> {
    if !terrain.sampling_ready() {
        return None;
    }
    let (before, after) = terrain.base.captures();
    let (b, a) = (before?, after?);
    if b.view != a.view
        || b.frame != a.frame
        || !b.taa_present
        || !a.taa_present
        || b.reset != expected_reset
        || a.reset != expected_reset
        || b.texture.id() == a.texture.id()
    {
        return None;
    }
    let world = terrain.base.render_world();
    let view = world.get::<ExtractedView>(a.view)?;
    if view.viewport != UVec4::new(0, 0, 64, 64) {
        return None;
    }
    let jitter = world.get::<TemporalJitter>(a.view)?;
    let size = UVec2::new(view.viewport.z, view.viewport.w);
    let view_from_world = view.world_from_view.to_matrix().inverse();
    let unjittered = view.clip_from_view * view_from_world;
    let mut projection = view.clip_from_view;
    jitter.jitter_projection(&mut projection, size.as_vec2());
    let raster = projection * view_from_world;
    let camera = view.world_from_view.translation();
    if !unjittered.is_finite() || !raster.is_finite() || !camera.is_finite() {
        return None;
    }
    Some(CameraFrame {
        view: a.view,
        sequence: b.frame,
        unjittered,
        raster,
        camera,
        before: b,
        after: a,
    })
}

fn require_frame(
    terrain: &crate::aa::bevy_reference::terrain::TerrainReferenceFixture,
    label: &str,
) -> CameraFrame {
    observed_frame(terrain, false).unwrap_or_else(|| {
        let (b, a) = terrain.base.captures();
        panic!(
            "{label}: no executed non-reset native TAA; sampling={} before={:?} after={:?}",
            terrain.sampling_ready(),
            b.map(|c| (c.view, c.frame, c.reset, c.taa_present, c.texture.id())),
            a.map(|c| (c.view, c.frame, c.reset, c.taa_present, c.texture.id()))
        )
    })
}

fn assert_case(
    fixture: &WindReferenceFixture,
    gpu: &super::gpu_reference::WindGpuReference,
    previous: &CameraFrame,
    current_camera: Vec3,
    previous_camera: Vec3,
    wrong_camera: Vec3,
    wind: &MonkeyFrame,
    require_sep: bool,
    require_zero: bool,
    label: &str,
) -> CameraFrame {
    assert!(
        gpu.is_ready(fixture.retained.terrain.base.render_world()),
        "{label}: gpu reference not ready"
    );
    let current = require_frame(&fixture.retained.terrain, label);
    assert_eq!(current.view, previous.view, "{label}: view changed");
    assert_eq!(
        current.sequence,
        previous.sequence + 1,
        "{label}: frame not consecutive"
    );
    assert!(
        (current.camera - current_camera).length() < 1.0e-4,
        "{label}: camera {:?}",
        current.camera
    );
    let cverts = gpu.evaluate(fixture, wind, current_camera);
    let pverts = gpu.evaluate(fixture, wind, previous_camera);
    let wverts = gpu.evaluate(fixture, wind, wrong_camera);
    let samples = camera_motion_samples(
        cverts,
        pverts,
        wverts,
        current.raster,
        current.unjittered,
        previous.unjittered,
    );
    let mut tri = [0usize; 2];
    let mut max_sep = 0.0_f32;
    for s in &samples {
        tri[s.triangle] += 1;
        max_sep = max_sep.max((s.expected - s.wrong_previous).abs().max_element());
    }
    assert!(
        samples.len() >= 4 && tri[0] > 0 && tri[1] > 0,
        "{label}: coverage samples={} tri={tri:?}",
        samples.len()
    );
    if require_sep {
        assert!(max_sep > 3.0e-3, "{label}: separation {max_sep}");
    }
    if require_zero {
        assert!(
            samples
                .iter()
                .all(|s| s.expected.abs().max_element() <= 1.0e-6),
            "{label}: nonzero expected"
        );
    }
    let source = fixture
        .retained
        .terrain
        .base
        .read_texture(&current.before.texture);
    let depth = fixture.retained.terrain.read_depth(&current.after);
    let prepass = fixture
        .retained
        .terrain
        .base
        .render_world()
        .get::<ViewPrepassTextures>(current.view)
        .expect("prepass");
    let tex = prepass
        .motion_vectors
        .as_ref()
        .expect("motion")
        .texture
        .texture
        .clone();
    let motion = fixture.retained.terrain.base.read_texture(&tex);
    let mut max_err = 0.0_f32;
    for s in &samples {
        let p = source[s.index];
        assert!(
            (p[0] - 1.0).abs() < 0.03 && p[1].abs() < 0.03 && (p[2] - 1.0).abs() < 0.03,
            "{label} frame={} pixel={:?} source={p:?}",
            current.sequence,
            s.pixel
        );
        assert!(
            depth[s.index][0] > 0.0 && depth[s.index][0].is_finite(),
            "{label} frame={} pixel={:?} depth",
            current.sequence,
            s.pixel
        );
        let actual = Vec2::new(motion[s.index][0], motion[s.index][1]);
        assert!(
            actual.is_finite() && s.expected.is_finite(),
            "{label} finite {:?}",
            s.pixel
        );
        if require_zero {
            assert!(
                actual.abs().max_element() <= 1.0e-3,
                "{label} frame={} pixel={:?} stationary actual={actual:?}",
                current.sequence,
                s.pixel
            );
        }
        let err = (actual - s.expected).abs();
        assert!(err.x<=1.0e-3 && err.y<=1.0e-3, "{label} frame={} view={:?} pixel={:?} actual={actual:?} expected={:?} err={err:?} reset={}/{}", current.sequence,current.view,s.pixel,s.expected,current.before.reset,current.after.reset);
        max_err = max_err.max(err.x).max(err.y);
    }
    println!("PASS wind {label}: frame={} samples={} triangles={tri:?} max_xy_error={max_err:.6} separation={max_sep:.6}", current.sequence, samples.len());
    current
}

pub(super) fn camera_discrimination_check() {
    let a = Vec3::new(0.0, 2.0, 2.0);
    let b = Vec3::new(0.0, 2.0, 5.0);
    let target = Vec3::new(0.0, 2.0, 0.0);
    let wind = frame();
    let projection = Mat4::perspective_infinite_reverse_rh(60.0_f32.to_radians(), 1.0, 0.1);
    let current = vertices(b, &wind);
    let previous = vertices(a, &wind);
    let wrong = vertices(b, &wind);
    let current_unj = projection * look(b, target);
    let previous_unj = projection * look(a, target);
    let samples = camera_motion_samples(
        current,
        previous,
        wrong,
        current_unj,
        current_unj,
        previous_unj,
    );
    let mut tri = [0usize; 2];
    let mut max_sep = 0.0_f32;
    for s in &samples {
        tri[s.triangle] += 1;
        max_sep = max_sep.max((s.expected - s.wrong_previous).abs().max_element());
    }
    assert!(samples.len() >= 4);
    assert!(tri[0] > 0 && tri[1] > 0);
    assert!(max_sep > 3.0e-3, "max separation={max_sep}");
    println!(
        "camera-discrimination samples={} triangles={tri:?} max_sep={max_sep:.6}",
        samples.len()
    );
}

pub(super) fn gpu_camera_check() {
    let a = Vec3::new(0.0, 2.0, 2.0);
    let b = Vec3::new(0.0, 2.0, 5.0);
    let target = Vec3::new(0.0, 2.0, 0.0);
    let wind = frame();
    let mut fixture = WindReferenceFixture::new_with_anchor(Vec3::ZERO);
    fixture.set_camera(a, target);
    fixture.set_wind(wind);
    let gpu = super::gpu_reference::WindGpuReference::new(&mut fixture.retained.terrain.base.app);
    fixture.retained.terrain.base.request_reset();
    let warm = fixture.retained.wait("camera warm", |terrain| {
        if !gpu.is_ready(terrain.base.render_world()) {
            return None;
        }
        let f = observed_frame(terrain, true)?;
        let src = terrain.base.read_texture(&f.before.texture);
        let dep = terrain.read_depth(&f.after);
        src.iter()
            .zip(dep.iter())
            .any(|(p, d)| {
                (p[0] - 1.0).abs() < 0.03
                    && p[1].abs() < 0.03
                    && (p[2] - 1.0).abs() < 0.03
                    && d[0] > 0.0
                    && d[0].is_finite()
            })
            .then_some(f)
    });
    println!(
        "PASS camera warm: frame={} view={:?}",
        warm.sequence, warm.view
    );
    fixture.retained.terrain.base.release_reset();
    let settled_a = fixture.retained.wait("camera steady A", |terrain| {
        if !gpu.is_ready(terrain.base.render_world()) {
            return None;
        }
        observed_frame(terrain, false)
    });
    fixture.update();
    let stationary_a = assert_case(
        &fixture,
        &gpu,
        &settled_a,
        a,
        a,
        a,
        &wind,
        false,
        true,
        "stationary A",
    );
    fixture.set_camera(b, target);
    fixture.update();
    let camera_b = assert_case(
        &fixture,
        &gpu,
        &stationary_a,
        b,
        a,
        b,
        &wind,
        true,
        false,
        "camera A->B",
    );
    fixture.update();
    let b1 = assert_case(
        &fixture,
        &gpu,
        &camera_b,
        b,
        b,
        b,
        &wind,
        false,
        true,
        "stationary B1",
    );
    fixture.update();
    let _b2 = assert_case(
        &fixture,
        &gpu,
        &b1,
        b,
        b,
        b,
        &wind,
        false,
        true,
        "stationary B2",
    );
    println!("EARLY-WIND CAMERA PASS");
}

#[cfg(test)]
mod tests;
