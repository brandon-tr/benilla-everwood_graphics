use super::*;
use crate::lighting::GxRetainedWindFrame;
use bevy::diagnostic::FrameCount;
use bevy::render::camera::TemporalJitter;

#[test]
fn tree_offset_is_frozen_and_changes_with_travel() {
    let mut frame = MonkeyFrame {
        wind_dir: [0.6, 0.8],
        wind_base_heading: 0.7,
        wind_gust: 0.5,
        wind_travel: 100.0,
        sway_strength: 1.0,
        tree_strength: 1.0,
        ..Default::default()
    };
    let world = Vec3::new(0.2, 2.5, 0.0);
    let anchor = Vec3::ZERO;
    let camera = Vec3::new(0.0, 2.0, 5.0);
    let a = tree_offset_reference(world, anchor, camera, &frame);
    let b = tree_offset_reference(world, anchor, camera, &frame);
    assert_eq!(a, b);
    assert!(a.length() > 0.0);
    frame.wind_travel += 50.0;
    assert_ne!(a, tree_offset_reference(world, anchor, camera, &frame));
}

#[test]
#[ignore = "Requires a GPU; run explicitly on the host PC"]
fn gpu_retained_native_wind_readback() {
    run_wind_case(WindReferenceFixture::new(), true);
}

#[test]
#[ignore = "Requires a GPU; run explicitly on the host PC"]
fn gpu_retained_native_wind_zero_phase_readback() {
    run_wind_case(WindReferenceFixture::new_with_anchor(Vec3::ZERO), false);
}

fn run_wind_case(mut fixture: WindReferenceFixture, use_gpu_reference: bool) {
    use crate::static_gx::RetainedTemporalSource;
    use bevy::core_pipeline::prepass::{
        ViewPrepassTextures, MOTION_VECTOR_PREPASS_FORMAT, NORMAL_PREPASS_FORMAT,
    };

    let gpu = use_gpu_reference.then(|| {
        super::gpu_reference::WindGpuReference::new(&mut fixture.retained.terrain.base.app)
    });
    let w0 = MonkeyFrame {
        wind_dir: [0.6, 0.8],
        wind_base_heading: 0.7,
        wind_gust: 0.5,
        wind_travel: 100.0,
        sway_strength: 1.0,
        tree_strength: 1.0,
        ..Default::default()
    };
    fixture.set_wind(w0);
    fixture.update();
    fixture.retained.terrain.base.request_reset();
    fixture.retained.wait("wind bootstrap", |terrain| {
        if !terrain.sampling_ready() {
            return None;
        }
        if let Some(gpu) = gpu.as_ref() {
            if !gpu.is_ready(terrain.base.render_world()) {
                return None;
            }
        }
        let (before, after) = terrain.base.captures();
        let (before, after) = (before?, after?);
        if before.view != after.view
            || before.frame != after.frame
            || !before.taa_present
            || !after.taa_present
            || !before.reset
            || !after.reset
            || before.texture.id() == after.texture.id()
        {
            return None;
        }
        let world = terrain.base.render_world();
        let prepass = world.get::<ViewPrepassTextures>(before.view)?;
        let depth = terrain.read_depth(&after);
        let normal_tex = prepass.normal.as_ref()?.texture.texture.clone();
        let motion_tex = prepass.motion_vectors.as_ref()?.texture.texture.clone();
        if normal_tex.format() != NORMAL_PREPASS_FORMAT
            || motion_tex.format() != MOTION_VECTOR_PREPASS_FORMAT
            || normal_tex.sample_count() != 1
            || motion_tex.sample_count() != 1
            || normal_tex.width() != 64
            || normal_tex.height() != 64
            || motion_tex.width() != 64
            || motion_tex.height() != 64
        {
            return None;
        }
        let source = terrain.base.read_texture(&before.texture);
        let normal = terrain.base.read_texture(&normal_tex);
        let motion = terrain.base.read_texture(&motion_tex);
        let mut published = false;
        world
            .resource::<RetainedTemporalSource>()
            .visit_temporal_wind(|r| {
                published = r.is_some();
                false
            });
        if !published {
            return None;
        }
        for i in 0..64 * 64 {
            let p = source[i];
            let d = depth[i][0];
            let n = normal[i];
            let m = motion[i];
            if p[3] > 0.99
                && (p[0] - 1.0).abs() < 0.03
                && p[1].abs() < 0.03
                && (p[2] - 1.0).abs() < 0.03
                && d > 0.0
                && d.is_finite()
                && (n[0] - 0.5).abs() < 0.03
                && (n[1] - 0.5).abs() < 0.03
                && (n[2] - 1.0).abs() < 0.03
                && n[3] > 0.99
                && m[0].is_finite()
                && m[1].is_finite()
            {
                return Some(());
            }
        }
        None
    });
    println!("PASS wind bootstrap: real classified color/depth/normal and native TAA");

    fixture.retained.terrain.base.release_reset();
    fixture.retained.wait("wind steady non-reset", |terrain| {
        let (before, after) = terrain.base.captures();
        let (before, after) = (before?, after?);
        (!before.reset && !after.reset && before.texture.id() != after.texture.id()).then_some(())
    });
    let (_, settled) = fixture.retained.terrain.base.captures();
    let settled = settled.expect("settled after");
    let f0 = settled.frame;
    let m0 = current_unjittered(&fixture, &settled);
    fixture.set_wind(w0);
    fixture.update();
    let (_, frozen) = fixture.retained.terrain.base.captures();
    let frozen = frozen.unwrap();
    let frozen_frame = frozen.frame;
    assert_eq!(frozen_frame, f0 + 1, "frozen consecutive");
    let (m0, d0) = assert_wind_motion_frame(
        &fixture,
        &w0,
        &w0,
        m0,
        None,
        false,
        "frozen W0",
        gpu.as_ref(),
    );
    let mut diagnostics: Vec<String> = d0.into_iter().collect();
    let w1 = MonkeyFrame {
        wind_travel: 150.0,
        ..w0
    };
    fixture.set_wind(w1);
    fixture.update();
    let (_, a1) = fixture.retained.terrain.base.captures();
    assert_eq!(a1.unwrap().frame, frozen_frame + 1, "W0->W1 consecutive");
    let (m1, d1) =
        assert_wind_motion_frame(&fixture, &w0, &w1, m0, None, true, "W0->W1", gpu.as_ref());
    diagnostics.extend(d1);
    let w2 = MonkeyFrame {
        wind_travel: 230.0,
        ..w0
    };
    fixture.set_wind(w2);
    fixture.update();
    let (_, a2) = fixture.retained.terrain.base.captures();
    assert_eq!(a2.unwrap().frame, frozen_frame + 2, "W1->W2 consecutive");
    let (_m2, d2) = assert_wind_motion_frame(
        &fixture,
        &w1,
        &w2,
        m1,
        Some(&w0),
        true,
        "W1->W2",
        gpu.as_ref(),
    );
    diagnostics.extend(d2);
    assert!(
        diagnostics.is_empty(),
        "wind motion diagnostics:\n{}",
        diagnostics.join("\n")
    );
}

fn current_unjittered(
    fixture: &WindReferenceFixture,
    capture: &crate::aa::bevy_reference::ObservedCapture,
) -> Mat4 {
    use bevy::render::view::ExtractedView;
    let base = &fixture.retained.terrain.base;
    let view = base
        .render_world()
        .get::<ExtractedView>(capture.view)
        .expect("view");
    view.clip_from_view * view.world_from_view.to_matrix().inverse()
}

fn assert_wind_motion_frame(
    fixture: &WindReferenceFixture,
    previous: &MonkeyFrame,
    current: &MonkeyFrame,
    previous_unjittered: Mat4,
    stale_previous: Option<&MonkeyFrame>,
    require_nonzero: bool,
    label: &str,
    gpu: Option<&super::gpu_reference::WindGpuReference>,
) -> (Mat4, Option<String>) {
    use bevy::core_pipeline::prepass::ViewPrepassTextures;
    use bevy::render::camera::TemporalJitter;
    use bevy::render::view::ExtractedView;

    let base = &fixture.retained.terrain;
    let (before, after) = base.base.captures();
    let (before, after) = (before.expect("before"), after.expect("after"));
    assert_eq!(before.view, after.view, "{label}");
    assert_eq!(before.frame, after.frame, "{label}");
    assert!(before.taa_present && after.taa_present, "{label}");
    assert!(!before.reset && !after.reset, "{label}");
    assert_ne!(before.texture.id(), after.texture.id(), "{label}");
    let world = base.base.render_world();
    let view = world.get::<ExtractedView>(after.view).expect("view");
    assert_eq!(view.viewport, UVec4::new(0, 0, 64, 64), "{label}");
    let jitter = world.get::<TemporalJitter>(after.view).expect("jitter");
    let inputs = wind_view_inputs(
        view.clip_from_view,
        view.world_from_view.to_matrix(),
        view.viewport,
        jitter,
    );
    assert_eq!(inputs.size, UVec2::splat(64), "{label}");
    assert!(
        inputs.unjittered.is_finite() && inputs.raster.is_finite(),
        "{label}"
    );
    assert!(
        (inputs.camera - Vec3::new(0.0, 2.0, 5.0)).length() < 1.0e-6,
        "{label}"
    );
    let viewport = inputs.size;
    let unjittered = inputs.unjittered;
    let raster = inputs.raster;
    let camera = inputs.camera;
    let anchor = fixture.anchor;
    let packed_current = current.pack(0.5, 0.0);
    let packed_previous = previous.pack(0.5, 0.0);
    let actual_rows = fixture
        .retained
        .terrain
        .base
        .app
        .world()
        .resource::<GxRetainedWindFrame>();
    assert_eq!(
        actual_rows.rows[4], packed_current[4],
        "{label}: current wind_a"
    );
    assert_eq!(
        actual_rows.rows[5], packed_current[5],
        "{label}: current wind_b"
    );
    println!(
        "{label} frame={} FrameCount={} view={:?} jitter={:?}",
        before.frame,
        fixture
            .retained
            .terrain
            .base
            .app
            .world()
            .resource::<FrameCount>()
            .0,
        before.view,
        jitter.offset
    );
    println!(
        "{label} MAIN-WORLD current_rows={:?}/{:?} previous_rows={:?}/{:?} MAIN-WORLD actual_rows={:?}/{:?}",
        packed_current[4],
        packed_current[5],
        packed_previous[4],
        packed_previous[5],
        actual_rows.rows[4],
        actual_rows.rows[5]
    );
    println!("{label} raster={raster:?} unjittered={unjittered:?} previous={previous_unjittered:?} camera={camera:?} anchor={anchor:?}");
    let anchor = fixture.anchor;
    let mut cur = [Vec3::ZERO; 4];
    let mut prev = [Vec3::ZERO; 4];
    let mut stale = [Vec3::ZERO; 4];
    for i in 0..4 {
        let p = TARGET_WORLD[i];
        cur[i] = p + tree_offset_reference(p, anchor, camera, current);
        prev[i] = p + tree_offset_reference(p, anchor, camera, previous);
        stale[i] = p + tree_offset_reference(p, anchor, camera, stale_previous.unwrap_or(previous));
    }
    println!(
        "{label} CPU current_deformed={cur:?} previous_deformed={prev:?} stale_deformed={stale:?}"
    );
    let (cur, prev, stale) = if let Some(gpu) = gpu {
        let gc = gpu.evaluate(fixture, current, camera);
        let gp = gpu.evaluate(fixture, previous, camera);
        let gs = gpu.evaluate(fixture, stale_previous.unwrap_or(previous), camera);
        println!("{label} GPU-reference current={gc:?} previous={gp:?} stale={gs:?}");
        (gc, gp, gs)
    } else {
        (cur, prev, stale)
    };
    let uvs = [Vec2::ZERO, Vec2::X, Vec2::ONE, Vec2::Y];
    let mut records: Vec<(usize, usize, usize, Vec2, Option<Vec2>, Vec3)> = Vec::new();
    let mut seen = [false; 4096];
    let mut per_triangle = [0usize; 2];
    for y in 0..64u32 {
        for x in 0..64u32 {
            let pixel = Vec2::new(x as f32 + 0.5, y as f32 + 0.5);
            let index = (y * 64 + x) as usize;
            if seen[index] {
                continue;
            }
            for (ti, tri) in [[0usize, 1, 2], [0, 2, 3]].into_iter().enumerate() {
                let cw = [cur[tri[0]], cur[tri[1]], cur[tri[2]]];
                let pw = [prev[tri[0]], prev[tri[1]], prev[tri[2]]];
                let Some(s) = expected_triangle_motion(
                    pixel,
                    viewport,
                    cw,
                    pw,
                    raster,
                    unjittered,
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
                let stale_motion = if stale_previous.is_some() {
                    let sw = [stale[tri[0]], stale[tri[1]], stale[tri[2]]];
                    expected_triangle_motion(
                        pixel,
                        viewport,
                        cw,
                        sw,
                        raster,
                        unjittered,
                        previous_unjittered,
                    )
                    .map(|s| s.motion)
                } else {
                    None
                };
                seen[index] = true;
                per_triangle[ti] += 1;
                records.push((index, ti, tri[0], s.motion, stale_motion, s.current_world));
                break;
            }
        }
    }
    assert!(records.len() >= 4, "{label}: samples={}", records.len());
    assert!(
        per_triangle[0] > 0 && per_triangle[1] > 0,
        "{label}: triangles={per_triangle:?}"
    );
    if require_nonzero {
        let nonzero = records
            .iter()
            .any(|r| r.3.x.abs() > 3.0e-3 || r.3.y.abs() > 3.0e-3);
        assert!(nonzero, "{label}: no expected component >3e-3");
    }
    if stale_previous.is_some() {
        let separated = records
            .iter()
            .any(|r| r.4.is_some_and(|s| (r.3 - s).abs().max_element() > 3.0e-3));
        assert!(separated, "{label}: stale counterfactual not separated");
    }
    let source = base.base.read_texture(&before.texture);
    let depth = base.read_depth(&after);
    let prepass = world
        .get::<ViewPrepassTextures>(after.view)
        .expect("prepass");
    let motion_tex = prepass
        .motion_vectors
        .as_ref()
        .expect("motion")
        .texture
        .texture
        .clone();
    let motion = base.base.read_texture(&motion_tex);
    let sample_count = records.len();
    let mut max_err = Vec2::ZERO;
    let mut violations = 0usize;
    let mut per_tri_violations = [0usize; 2];
    let mut signed_min = Vec2::splat(f32::INFINITY);
    let mut signed_max = Vec2::splat(f32::NEG_INFINITY);
    let mut reps: Vec<String> = Vec::new();
    let mut worst = String::new();
    let mut worst_err = 0.0_f32;
    let mut max_depth_diff = 0.0_f32;
    let mut depth_pixel = 0usize;
    let mut max_stale = (0.0_f32, 0usize);
    for (index, ti, _tri, expected, stale_motion, current_world) in records {
        let p = source[index];
        assert!(
            (p[0] - 1.0).abs() < 0.03 && p[1].abs() < 0.03 && (p[2] - 1.0).abs() < 0.03,
            "{label}: source idx={index}"
        );
        assert!(
            depth[index][0] > 0.0 && depth[index][0].is_finite(),
            "{label}: depth idx={index}"
        );
        let actual = Vec2::new(motion[index][0], motion[index][1]);
        assert!(
            actual.is_finite() && expected.is_finite(),
            "{label}: nonfinite idx={index}"
        );
        let err = (actual - expected).abs();
        max_err = max_err.max(err);
        signed_min = signed_min.min(actual - expected);
        signed_max = signed_max.max(actual - expected);
        let clip = unjittered * current_world.extend(1.0);
        let expected_depth = clip.z / clip.w;
        let depth_diff = (expected_depth - depth[index][0]).abs();
        if depth_diff > max_depth_diff {
            max_depth_diff = depth_diff;
            depth_pixel = index;
        }
        if let Some(stale) = stale_motion {
            let separation = (expected - stale).abs().max_element();
            if separation > max_stale.0 {
                max_stale = (separation, index);
            }
        }
        if err.x > 1.0e-3 || err.y > 1.0e-3 {
            violations += 1;
            per_tri_violations[ti] += 1;
            let msg = format!(
                "frame={} idx={index} tri={ti} actual={actual:?} expected={expected:?} err={err:?}",
                before.frame
            );
            if reps.len() < 8 {
                reps.push(msg.clone());
            }
            if err.max_element() > worst_err {
                worst_err = err.max_element();
                worst = msg;
            }
        }
    }
    let diag = if violations > 0 {
        let summary = format!(
            "FAIL/DIAGNOSTIC wind {label}: samples={sample_count} triangles={per_triangle:?} violations={violations} per_triangle={per_tri_violations:?} max_abs=({:.9},{:.9}) signed_min=({:.9},{:.9}) signed_max=({:.9},{:.9}) depth_max_abs={max_depth_diff:.9} depth_pixel={depth_pixel} stale_max={:.9} stale_pixel={} worst=[{worst}] reps=[{}]",
            max_err.x, max_err.y, signed_min.x, signed_min.y, signed_max.x, signed_max.y, max_stale.0, max_stale.1, reps.join("; ")
        );
        println!("{summary}");
        Some(summary)
    } else {
        println!("PASS wind {label}: samples={sample_count} triangles={per_triangle:?} max_xy_error={:.6} depth_max_abs={max_depth_diff:.9} depth_pixel={depth_pixel} stale_max={:.9} stale_pixel={}", max_err.max_element(), max_stale.0, max_stale.1);
        None
    };
    (unjittered, diag)
}
#[test]
fn fixture_normal_round_trips_to_bevy_plus_z() {
    let mesh = tree_geometry();
    for n in &mesh.normals {
        let bevy = benilla_assets::coords::wow_to_bevy(*n);
        assert!((bevy - Vec3::Z).length() < 1.0e-6);
    }
}

#[test]
fn wgsl_fract_matches_negative_floor_semantics() {
    assert!((wgsl_fract(-0.25) - 0.75).abs() < 1.0e-6);
    assert!((wgsl_fract(1.25) - 0.25).abs() < 1.0e-6);
}

fn test_triangle() -> ([Vec3; 3], Mat4) {
    (
        [
            Vec3::new(-1.0, 0.0, -2.0),
            Vec3::new(1.0, 0.0, -2.0),
            Vec3::new(0.0, 1.0, -3.0),
        ],
        Mat4::perspective_infinite_reverse_rh(1.0, 1.0, 0.1),
    )
}

#[test]
fn perspective_oracle_matches_zero_and_rejects_outside() {
    let (world, m) = test_triangle();
    let mut screen = [Vec2::ZERO; 3];
    for i in 0..3 {
        let c = m * world[i].extend(1.0);
        let ndc = c.xy() / c.w;
        screen[i] = Vec2::new((ndc.x * 0.5 + 0.5) * 64.0, (0.5 - ndc.y * 0.5) * 64.0);
    }
    let pixel = (screen[0] + screen[1] + screen[2]) / 3.0;
    let s =
        expected_triangle_motion(pixel, UVec2::splat(64), world, world, m, m, m).expect("inside");
    assert!((s.weights.iter().sum::<f32>() - 1.0).abs() < 1.0e-5);
    assert!((s.motion - Vec2::ZERO).length() < 1.0e-6);
    assert!(expected_triangle_motion(
        Vec2::new(-100.0, -100.0),
        UVec2::splat(64),
        world,
        world,
        m,
        m,
        m
    )
    .is_none());
    assert!(expected_triangle_motion(
        Vec2::splat(32.0),
        UVec2::splat(64),
        [Vec3::ZERO; 3],
        [Vec3::ZERO; 3],
        m,
        m,
        m
    )
    .is_none());
}

#[test]
fn unequal_w_uses_perspective_interpolation() {
    let (world, m) = test_triangle();
    let screen = projected_screen(world, m);
    let pixel = (screen[0] + screen[1] + screen[2]) / 3.0;
    let s =
        expected_triangle_motion(pixel, UVec2::splat(64), world, world, m, m, m).expect("inside");
    for (got, want) in s.weights.iter().zip([3.0 / 8.0, 3.0 / 8.0, 1.0 / 4.0]) {
        assert!((got - want).abs() < 1.0e-5);
    }
    let expected = Vec3::new(0.0, 0.25, -2.25);
    assert!((s.current_world - expected).length() < 1.0e-5);
    assert!((s.previous_world - expected).length() < 1.0e-5);
    assert!((s.current_world - Vec3::new(0.0, 1.0 / 3.0, -7.0 / 3.0)).length() > 1.0e-4);
}

#[test]
fn raster_only_jitter_does_not_create_motion() {
    let (world, m) = test_triangle();
    let raster = Mat4::from_translation(Vec3::new(0.1, 0.07, 0.0)) * m;
    let screen = projected_screen(world, raster);
    let pixel = (screen[0] + screen[1] + screen[2]) / 3.0;
    let s = expected_triangle_motion(pixel, UVec2::splat(64), world, world, raster, m, m)
        .expect("inside");
    assert!(s.motion.x.abs() < 1.0e-6 && s.motion.y.abs() < 1.0e-6);
    let un = projected_screen(world, m);
    let un_pixel = (un[0] + un[1] + un[2]) / 3.0;
    assert!((pixel - un_pixel).length() > 0.5);
}

#[test]
fn positive_x_camera_translation_has_negative_native_motion() {
    let world = [
        Vec3::new(-1.0, -1.0, -5.0),
        Vec3::new(1.0, -1.0, -5.0),
        Vec3::new(0.0, 1.0, -5.0),
    ];
    let p = Mat4::perspective_infinite_reverse_rh(60.0_f32.to_radians(), 1.0, 0.1);
    let current = p * Mat4::from_translation(Vec3::new(-0.1, 0.0, 0.0));
    let screen = projected_screen(world, current);
    let pixel = (screen[0] + screen[1] + screen[2]) / 3.0;
    let s = expected_triangle_motion(pixel, UVec2::splat(64), world, world, current, current, p)
        .expect("inside");
    let expected = -0.1 / (2.0 * 30.0_f32.to_radians().tan() * 5.0);
    assert!(s.motion.x < 0.0 && (s.motion.x - expected).abs() < 1.0e-6);
    assert!(s.motion.y.abs() < 1.0e-6);
}

fn projected_screen(world: [Vec3; 3], m: Mat4) -> [Vec2; 3] {
    let mut out = [Vec2::ZERO; 3];
    for i in 0..3 {
        let c = m * world[i].extend(1.0);
        let ndc = c.xy() / c.w;
        out[i] = Vec2::new((ndc.x * 0.5 + 0.5) * 64.0, (0.5 - ndc.y * 0.5) * 64.0);
    }
    out
}

struct WindViewInputs {
    size: UVec2,
    unjittered: Mat4,
    raster: Mat4,
    camera: Vec3,
}

fn wind_view_inputs(
    clip_from_view: Mat4,
    world_from_view: Mat4,
    viewport: UVec4,
    jitter: &TemporalJitter,
) -> WindViewInputs {
    let size = viewport.zw();
    let view_from_world = world_from_view.inverse();
    let unjittered = clip_from_view * view_from_world;
    let mut jittered_projection = clip_from_view;
    jitter.jitter_projection(&mut jittered_projection, size.as_vec2());
    WindViewInputs {
        size,
        unjittered,
        raster: jittered_projection * view_from_world,
        camera: world_from_view.transform_point3(Vec3::ZERO),
    }
}

#[test]
fn wind_view_inputs_use_viewport_dimensions_and_camera_translation() {
    use bevy::render::camera::TemporalJitter;
    let projection = Mat4::perspective_infinite_reverse_rh(60.0_f32.to_radians(), 1.0, 0.1);
    let world = Mat4::from_translation(Vec3::new(0.0, 2.0, 5.0));
    let jitter = TemporalJitter {
        offset: Vec2::new(0.25, -0.25),
    };
    let inputs = wind_view_inputs(projection, world, UVec4::new(0, 0, 64, 64), &jitter);
    assert_eq!(inputs.size, UVec2::splat(64));
    assert!((inputs.camera - Vec3::new(0.0, 2.0, 5.0)).length() < 1.0e-6);
    assert!(inputs.unjittered.is_finite() && inputs.raster.is_finite());
    assert_ne!(inputs.raster, inputs.unjittered);
    assert_eq!(inputs.unjittered, projection * world.inverse());
}

#[test]
fn negative_anchor_wind_is_finite() {
    let frame = MonkeyFrame {
        wind_dir: [0.6, 0.8],
        wind_base_heading: 0.7,
        wind_gust: 0.5,
        wind_travel: 100.0,
        sway_strength: 1.0,
        tree_strength: 1.0,
        ..Default::default()
    };
    let off = tree_offset_reference(
        Vec3::new(-2.0, 2.5, -3.0),
        Vec3::new(-2.0, 0.0, -3.0),
        Vec3::new(0.0, 2.0, 5.0),
        &frame,
    );
    assert!(off.is_finite() && off.length() > 0.0);
}

#[test]
fn anchor_control_preserves_undeformed_world_geometry() {
    use benilla_assets::coords::wow_to_bevy;
    for anchor in [Vec3::new(0.0, 0.0, 0.3), Vec3::ZERO] {
        let mesh = tree_geometry_at_anchor(anchor);
        for (i, p) in mesh.positions.iter().enumerate() {
            let world = wow_to_bevy(*p) + anchor;
            assert!(
                (world - TARGET_WORLD[i]).length() < 1.0e-6,
                "anchor={anchor:?}"
            );
        }
        for n in &mesh.normals {
            assert!((wow_to_bevy(*n) - Vec3::Z).length() < 1.0e-6);
        }
    }
    let dot = Vec2::ZERO.dot(Vec2::new(0.173, 0.317));
    assert_eq!(dot, 0.0);
    assert_eq!(wgsl_fract(dot.sin() * 43_758.547), 0.0);
}

#[test]
fn gpu_reference_input_rows_match_uniform_layout() {
    use super::gpu_reference::{reference_input_rows, INPUT_BYTES};
    let frame = MonkeyFrame {
        wind_dir: [0.25, 0.75],
        wind_base_heading: 1.5,
        wind_gust: 0.85,
        wind_travel: 177.0,
        sway_strength: 0.65,
        tree_strength: 0.95,
        ..Default::default()
    };
    let anchor = Vec3::new(1.5, -2.5, 3.5);
    let camera = Vec3::new(-4.0, 6.0, 7.0);
    let rows = reference_input_rows(anchor, camera, &frame);
    assert_eq!(INPUT_BYTES, 128);
    assert_eq!(std::mem::size_of_val(&rows) as u64, 128);
    for i in 0..4 {
        assert_eq!(
            rows[i],
            [TARGET_WORLD[i].x, TARGET_WORLD[i].y, TARGET_WORLD[i].z, 1.0]
        );
    }
    assert_eq!(rows[4], [1.5, -2.5, 3.5, 1.0]);
    assert_eq!(rows[5], [-4.0, 6.0, 7.0, 1.0]);
    let packed = frame.pack(0.5, 0.0);
    assert_eq!(rows[6], packed[4]);
    assert_eq!(rows[7], packed[5]);
}
