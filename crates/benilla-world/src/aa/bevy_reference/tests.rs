use bevy::color::LinearRgba;
use bevy::core_pipeline::prepass::MOTION_VECTOR_PREPASS_FORMAT;
use bevy::render::render_resource::TextureFormat;

use super::*;

fn expected_linear() -> [f32; 3] {
    let linear = LinearRgba::from(Color::srgb(0.2, 0.6, 0.9));
    [linear.red, linear.green, linear.blue]
}

fn is_approx(actual: f32, expected: f32, tolerance: f32) -> bool {
    (actual - expected).abs() <= tolerance
}

fn pixel(pixels: &[[f32; 4]], x: u32, y: u32) -> [f32; 4] {
    pixels[(y * SIZE.x + x) as usize]
}

fn finite(rgba: [f32; 4]) -> bool {
    rgba.iter().all(|value| value.is_finite())
}

fn pair(fixture: &BevyReferenceFixture) -> (ObservedCapture, ObservedCapture) {
    let (before, after) = fixture.captures();
    let before = before.expect("before-TAA capture");
    let after = after.expect("after-TAA capture");
    assert_eq!(
        before.view, after.view,
        "before/after must be the same view"
    );
    assert_eq!(
        before.frame, after.frame,
        "before/after must be the same frame"
    );
    (before, after)
}

fn executed_pair(fixture: &BevyReferenceFixture) -> (ObservedCapture, ObservedCapture) {
    let (before, after) = pair(fixture);
    assert!(before.taa_present, "TemporalAntiAliasing must be present");
    assert_ne!(
        before.texture.id(),
        after.texture.id(),
        "native TAA did not flip the ViewTarget texture"
    );
    (before, after)
}

fn compare_output_to_source(
    fixture: &BevyReferenceFixture,
    before: &ObservedCapture,
    after: &ObservedCapture,
) {
    let source = fixture.read_texture(&before.texture);
    let output = fixture.read_texture(&after.texture);
    for y in 0..SIZE.y {
        for x in 0..SIZE.x {
            let source_pixel = pixel(&source, x, y);
            let output_pixel = pixel(&output, x, y);
            assert!(finite(source_pixel), "non-finite source at {x},{y}");
            assert!(finite(output_pixel), "non-finite TAA output at {x},{y}");
            for channel in 0..3 {
                assert!(
                    is_approx(source_pixel[channel], output_pixel[channel], 8.0e-3),
                    "reset TAA changed {x},{y} channel {channel}: {:?} vs {:?}",
                    source_pixel,
                    output_pixel
                );
            }
        }
    }
}

fn warmup_reset(fixture: &mut BevyReferenceFixture) -> ObservedCapture {
    let expected = expected_linear();
    wait_reference_stage(fixture, "reset TAA prerequisite", |fixture| {
        if !fixture.sampling_ready() {
            return None;
        }
        let (before, after) = fixture.captures();
        let (Some(before), Some(after)) = (before, after) else {
            return None;
        };
        if before.view != after.view
            || before.frame != after.frame
            || !before.taa_present
            || !before.reset
            || before.texture.id() == after.texture.id()
        {
            return None;
        }
        let (Some(depth), Some(motion)) = (before.depth, before.motion.as_ref()) else {
            return None;
        };
        if depth.1.width != SIZE.x
            || depth.1.height != SIZE.y
            || motion.1.width != SIZE.x
            || motion.1.height != SIZE.y
        {
            return None;
        }
        let source = fixture.read_texture(&before.texture);
        let output = fixture.read_texture(&after.texture);
        let center = pixel(&source, SIZE.x / 2, SIZE.y / 2);
        let corner_a = pixel(&source, 2, 2);
        let corner_b = pixel(&source, SIZE.x - 3, SIZE.y - 3);
        let center_ok = (0..3).all(|channel| is_approx(center[channel], expected[channel], 8.0e-3));
        let corner_ok = corner_a[0..3].iter().all(|value| value.abs() < 0.02)
            && corner_b[0..3].iter().all(|value| value.abs() < 0.02);
        let output_ok = output.iter().flatten().all(|value| value.is_finite());
        (center_ok && corner_ok && output_ok).then_some(after)
    })
}

fn edge_at(source: &[[f32; 4]], x: u32, y: u32) -> bool {
    let here = pixel(source, x, y);
    for (dx, dy) in [(-1i32, 0i32), (1, 0), (0, -1), (0, 1)] {
        let nx = x as i32 + dx;
        let ny = y as i32 + dy;
        if nx < 0 || ny < 0 || nx >= SIZE.x as i32 || ny >= SIZE.y as i32 {
            continue;
        }
        let there = pixel(source, nx as u32, ny as u32);
        if (0..3).any(|channel| (here[channel] - there[channel]).abs() > 0.05) {
            return true;
        }
    }
    false
}

#[test]
#[ignore = "Requires a GPU; run explicitly on the host PC"]
fn gpu_bevy_taa_reference_readback() {
    let mut fixture = BevyReferenceFixture::new();
    fixture.print_adapter();
    let expected = expected_linear();

    // A. Actual reset pass.
    warmup_reset(&mut fixture);
    let (reset_before, reset_after) = pair(&fixture);
    assert_eq!(reset_before.frame, reset_after.frame);
    assert!(reset_before.reset && reset_after.reset);
    assert_ne!(reset_before.texture.id(), reset_after.texture.id());
    compare_output_to_source(&fixture, &reset_before, &reset_after);
    let reset_source = fixture.read_texture(&reset_before.texture);
    let center = pixel(&reset_source, SIZE.x / 2, SIZE.y / 2);
    for channel in 0..3 {
        assert!(is_approx(center[channel], expected[channel], 8.0e-3));
    }
    let (depth_format, depth_size) = reset_before.depth.expect("depth prepass");
    let (motion_format, motion_size, _) = reset_before.motion.clone().expect("motion prepass");
    assert_eq!(depth_format, TextureFormat::Depth32Float);
    assert_eq!(motion_format, MOTION_VECTOR_PREPASS_FORMAT);
    assert_eq!((depth_size.width, depth_size.height), (SIZE.x, SIZE.y));
    assert_eq!((motion_size.width, motion_size.height), (SIZE.x, SIZE.y));
    println!("PASS A: native reset TAA executed, passed through, and prepasses are 64x64");

    // B. Jitter and reconstruction.
    fixture.release_reset();
    let first_nonreset = wait_reference_stage(&mut fixture, "first non-reset TAA", |fixture| {
        if !fixture.sampling_ready() {
            return None;
        }
        let (before, after) = fixture.captures();
        let (Some(before), Some(after)) = (before, after) else {
            return None;
        };
        (!before.reset
            && !after.reset
            && before.view == after.view
            && before.frame == after.frame
            && before.taa_present
            && before.texture.id() != after.texture.id())
        .then_some(after)
    });
    assert!(!first_nonreset.reset);

    let mut jitters = Vec::new();
    let mut changed_edges = 0usize;
    let mut largest = 0.0f32;
    let mut fractional_edge = false;
    for frame in 0..16 {
        fixture.update();
        let (before, after) = executed_pair(&fixture);
        assert!(!before.reset, "frame {frame} unexpectedly reset");
        let source = fixture.read_texture(&before.texture);
        let output = fixture.read_texture(&after.texture);
        let jitter = before.jitter.expect("TemporalJitter is present");
        assert!(jitter.is_finite() && jitter.x.abs() <= 0.5 && jitter.y.abs() <= 0.5);
        jitters.push(jitter);
        for y in 0..SIZE.y {
            for x in 0..SIZE.x {
                let source_pixel = pixel(&source, x, y);
                let output_pixel = pixel(&output, x, y);
                assert!(finite(source_pixel) && finite(output_pixel));
                let difference = (0..3)
                    .map(|channel| (source_pixel[channel] - output_pixel[channel]).abs())
                    .fold(0.0f32, f32::max);
                largest = largest.max(difference);
                if edge_at(&source, x, y) && difference > 1.0e-3 {
                    changed_edges += 1;
                }
                if edge_at(&source, x, y)
                    && output_pixel[1] > 0.02
                    && output_pixel[1] < expected[1] - 0.02
                {
                    fractional_edge = true;
                }
            }
        }
        let (_, _, motion_texture) = before.motion.as_ref().expect("motion prepass");
        let motion = fixture.read_texture(motion_texture);
        let center_motion = pixel(&motion, SIZE.x / 2, SIZE.y / 2);
        assert!(center_motion[0].abs() < 1.0e-3 && center_motion[1].abs() < 1.0e-3);
    }
    let distinct_jitter = jitters
        .iter()
        .any(|a| jitters.iter().any(|b| a.distance(*b) > 1.0e-6));
    assert!(distinct_jitter, "jitter did not vary across frames");
    assert!(changed_edges > 0, "TAA did not change any edge pixel");
    assert!(fractional_edge, "no edge pixel showed fractional coverage");
    let jitter_min = jitters
        .iter()
        .copied()
        .fold(Vec2::splat(f32::INFINITY), Vec2::min);
    let jitter_max = jitters
        .iter()
        .copied()
        .fold(Vec2::splat(f32::NEG_INFINITY), Vec2::max);
    println!(
        "PASS B: 16 non-reset TAA frames, changed_edges={changed_edges}, largest_difference={largest}, jitter_min={jitter_min:?}, jitter_max={jitter_max:?}"
    );

    // C. Explicit camera-cut-style reset.
    // `reset_source` is immutable CPU pixels from case A. The ViewTarget texture handle only
    // retains the reused allocation, so it must not be reread as an old-frame snapshot after B.
    fixture.camera_cut(0.25);
    warmup_reset(&mut fixture);
    let (cut_before, cut_after) = pair(&fixture);
    assert_eq!(cut_before.frame, cut_after.frame);
    assert!(cut_before.reset && cut_after.reset);
    assert_ne!(cut_before.texture.id(), cut_after.texture.id());
    compare_output_to_source(&fixture, &cut_before, &cut_after);
    let cut_source = fixture.read_texture(&cut_before.texture);
    assert!(
        cut_source
            .iter()
            .zip(reset_source.iter())
            .any(|(a, b)| (0..3).any(|channel| (a[channel] - b[channel]).abs() > 1.0e-3)),
        "camera cut did not change the source scene"
    );
    println!("PASS C: explicit CameraCut reset matched its changed source");

    // D. Disable native TAA.
    fixture.disable_taa();
    {
        let world = fixture.app.world();
        assert!(world.get::<TemporalAntiAliasing>(fixture.camera).is_none());
        assert!(world.get::<TemporalJitter>(fixture.camera).is_none());
        assert!(world.get::<MipBias>(fixture.camera).is_none());
        assert!(world.get::<DepthPrepass>(fixture.camera).is_none());
        assert!(world.get::<MotionVectorPrepass>(fixture.camera).is_none());
    }

    let mut candidates = 0usize;
    let mut initial_center = None;
    let (disabled, disabled_pixels, first_center) = wait_reference_stage(
        &mut fixture,
        "disabled TAA frame",
        |fixture| {
            if !fixture.sampling_ready() {
                fixture.stage_evidence = "sampling pipeline not ready".into();
                return None;
            }
            let (before, after) = fixture.captures();
            let (Some(before), Some(after)) = (before, after) else {
                fixture.stage_evidence = "no paired disabled observers".into();
                return None;
            };
            let Some(render_entity) = fixture.render_entity() else {
                fixture.stage_evidence = format!(
                    "frame={} view={:?} render_entity=absent",
                    after.frame, after.view
                );
                return None;
            };
            let (taa_render_absent, jitter_render_absent) = {
                let world = fixture.render_world();
                (
                    world.get::<TemporalAntiAliasing>(render_entity).is_none(),
                    world.get::<TemporalJitter>(render_entity).is_none(),
                )
            };
            let lifecycle_ready = !before.taa_present
                && !after.taa_present
                && before.frame == after.frame
                && before.view == after.view
                && before.texture.id() == after.texture.id()
                && taa_render_absent
                && jitter_render_absent;
            if !lifecycle_ready {
                fixture.stage_evidence = format!(
                    "frame={} view={:?} before_id={:?} after_id={:?} taa_present={}/{} taa_render_absent={taa_render_absent} jitter_render_absent={jitter_render_absent} pixel_checks=deferred (disabled lifecycle not ready)",
                    after.frame,
                    after.view,
                    before.texture.id(),
                    after.texture.id(),
                    before.taa_present,
                    after.taa_present,
                );
                return None;
            }

            candidates += 1;
            let pixels = fixture.read_texture(&after.texture);
            let center = pixel(&pixels, SIZE.x / 2, SIZE.y / 2);
            if initial_center.is_none() {
                initial_center = Some(center);
            }
            let finite = pixels.iter().flatten().all(|value| value.is_finite());
            let center_ok =
                (0..3).all(|channel| is_approx(center[channel], expected[channel], 8.0e-3));
            fixture.stage_evidence = format!(
                "frame={} view={:?} before_id={:?} after_id={:?} taa_present={}/{} taa_render_absent={taa_render_absent} jitter_render_absent={jitter_render_absent} center={center:?} expected={expected:?} finite={finite} center_ok={center_ok}",
                after.frame,
                after.view,
                before.texture.id(),
                after.texture.id(),
                before.taa_present,
                after.taa_present,
            );
            (finite && center_ok).then_some((after, pixels, initial_center.unwrap()))
        },
    );

    let render_entity = fixture.render_entity().expect("render camera entity");
    let render_world = fixture.render_world();
    assert!(render_world
        .get::<TemporalAntiAliasing>(render_entity)
        .is_none());
    assert!(render_world.get::<TemporalJitter>(render_entity).is_none());
    assert!(disabled_pixels
        .iter()
        .flatten()
        .all(|value| value.is_finite()));
    let center = pixel(&disabled_pixels, SIZE.x / 2, SIZE.y / 2);
    assert!(
        (0..3).all(|channel| is_approx(center[channel], expected[channel], 8.0e-3)),
        "disabled scene center {center:?} does not match expected {expected:?}"
    );
    println!(
        "[RP03A-D] candidates={candidates} initial_center={first_center:?} latest_center={center:?} frame={}",
        disabled.frame
    );
    println!("PASS D: disabling native TAA removed sync components and stopped the texture flip");
}
