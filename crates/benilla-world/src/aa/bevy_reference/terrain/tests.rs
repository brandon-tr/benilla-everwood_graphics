use super::*;

use bevy::core_pipeline::prepass::{
    DepthPrepass, MotionVectorPrepass, MOTION_VECTOR_PREPASS_FORMAT,
};
use bevy::mesh::{Indices, VertexAttributeValues};
use bevy::render::render_resource::TextureFormat;

const SIZE: UVec2 = super::super::SIZE;
const EXPECTED: [f32; 3] = [51.0 / 255.0, 153.0 / 255.0, 230.0 / 255.0];

fn pixel(pixels: &[[f32; 4]], x: u32, y: u32) -> [f32; 4] {
    pixels[pixel_index(x, y)]
}

fn finite(pixel: [f32; 4]) -> bool {
    pixel.iter().all(|value| value.is_finite())
}

fn approx(a: f32, b: f32, tolerance: f32) -> bool {
    (a - b).abs() <= tolerance
}

fn color_visible(pixel: [f32; 4]) -> bool {
    pixel[0].max(pixel[1]).max(pixel[2]) > 0.01
}

fn edge_at(pixels: &[[f32; 4]], x: u32, y: u32) -> bool {
    let here = pixel(pixels, x, y);
    for (dx, dy) in [(-1i32, 0i32), (1, 0), (0, -1), (0, 1)] {
        let nx = x as i32 + dx;
        let ny = y as i32 + dy;
        if nx < 0 || ny < 0 || nx >= SIZE.x as i32 || ny >= SIZE.y as i32 {
            continue;
        }
        let there = pixel(pixels, nx as u32, ny as u32);
        if (0..3).any(|channel| (here[channel] - there[channel]).abs() > 0.05) {
            return true;
        }
    }
    false
}

fn motion_at(pixels: &[[f32; 4]], x: u32, y: u32) -> Vec2 {
    let p = pixel(pixels, x, y);
    Vec2::new(p[0], p[1])
}

fn taa_executed_pair(fixture: &TerrainReferenceFixture) -> (ObservedCapture, ObservedCapture) {
    let (before, after) = fixture.pair();
    assert!(before.taa_present && after.taa_present, "TAA must execute");
    assert_ne!(
        before.texture.id(),
        after.texture.id(),
        "TAA must flip the ViewTarget"
    );
    (before, after)
}

#[test]
fn terrain_mesh_fixture_has_expected_geometry() {
    let mesh = terrain_mesh().expect("actual chunk_to_mesh output");
    let positions = match mesh.attribute(Mesh::ATTRIBUTE_POSITION) {
        Some(VertexAttributeValues::Float32x3(values)) => values,
        other => panic!("terrain positions: {other:?}"),
    };
    let expected = [
        [-1.5, -1.5, 0.0],
        [1.5, -1.5, 0.0],
        [1.5, 1.5, 0.0],
        [-1.5, 1.5, 0.0],
    ];
    assert_eq!(positions, &expected);
    let normals = match mesh.attribute(Mesh::ATTRIBUTE_NORMAL) {
        Some(VertexAttributeValues::Float32x3(values)) => values,
        other => panic!("terrain normals: {other:?}"),
    };
    assert!(normals.iter().all(|normal| approx(normal[2], 1.0, 1e-6)));
    assert!(normals
        .iter()
        .all(|normal| normal[0].abs() < 1e-6 && normal[1].abs() < 1e-6));
    let indices = match mesh.indices() {
        Some(Indices::U32(indices)) => indices,
        other => panic!("terrain indices: {other:?}"),
    };
    assert_eq!(indices, &[0, 1, 2, 0, 2, 3]);
    let uv1 = match mesh.attribute(Mesh::ATTRIBUTE_UV_1) {
        Some(VertexAttributeValues::Float32x2(values)) => values,
        other => panic!("terrain uv1: {other:?}"),
    };
    assert!(uv1.iter().all(|uv| uv[0] == 0.0 && uv[1] == -1.0));
    let colors = match mesh.attribute(Mesh::ATTRIBUTE_COLOR) {
        Some(VertexAttributeValues::Float32x4(values)) => values,
        other => panic!("terrain colors: {other:?}"),
    };
    assert!(colors.iter().all(|color| color == &[0.0, 0.0, 0.0, 0.0]));
}

#[test]
#[ignore = "Requires a GPU; run explicitly on the host PC"]
fn gpu_terrain_material_native_taa_readback() {
    let mut fixture = TerrainReferenceFixture::new();
    fixture.print_adapter();

    // A. Real terrain color and native prepass coverage.
    let _warmup = fixture.wait("terrain native reset warmup", |fixture| {
        if !fixture.sampling_ready() {
            return None;
        }
        let (before, after) = fixture.base.captures();
        let (Some(before), Some(after)) = (before, after) else {
            return None;
        };
        if !before.reset
            || !after.reset
            || !before.taa_present
            || before.view != after.view
            || before.frame != after.frame
        {
            return None;
        }
        if before.texture.id() == after.texture.id() {
            return None;
        }
        let depth = fixture.depth_texture(&after)?;
        if depth.format() != TextureFormat::Depth32Float
            || depth.width() != SIZE.x
            || depth.height() != SIZE.y
        {
            return None;
        }
        let source = fixture.base.read_texture(&before.texture);
        let center = pixel(&source, SIZE.x / 2, SIZE.y / 2);
        fixture.last_pixels = format!("source_center={center:?} expected={EXPECTED:?}");
        if !color_visible(center) {
            return None;
        }
        let depth_pixels = fixture.read_depth(&after);
        fixture.last_pixels = format!(
            "source_center={center:?} depth_center={:?}",
            pixel(&depth_pixels, SIZE.x / 2, SIZE.y / 2)
        );
        if pixel(&depth_pixels, SIZE.x / 2, SIZE.y / 2)[0] <= 0.0 {
            return None;
        }
        fixture.base.stage_evidence = format!(
            "reset={} source_center={center:?} depth={:?}",
            before.reset,
            pixel(&depth_pixels, SIZE.x / 2, SIZE.y / 2)
        );
        Some((before, after))
    });
    let (reset_before, reset_after) = taa_executed_pair(&fixture);
    assert_eq!(reset_before.frame, reset_after.frame);
    assert!(reset_before.reset && reset_after.reset);
    assert_eq!(
        reset_before.depth.expect("native depth attachment").0,
        TextureFormat::Depth32Float
    );
    assert_eq!(
        reset_before.motion.as_ref().expect("native motion").0,
        MOTION_VECTOR_PREPASS_FORMAT
    );
    let depth_texture = fixture
        .depth_texture(&reset_before)
        .expect("native depth texture");
    assert_eq!(depth_texture.sample_count(), 1);
    assert_eq!(
        (depth_texture.width(), depth_texture.height()),
        (SIZE.x, SIZE.y)
    );
    let motion_texture = reset_before
        .motion
        .as_ref()
        .expect("native motion texture")
        .2
        .clone();
    assert_eq!(motion_texture.sample_count(), 1);
    assert_eq!(
        (motion_texture.width(), motion_texture.height()),
        (SIZE.x, SIZE.y)
    );
    let motion = fixture.base.read_texture(&motion_texture);
    let source = fixture.base.read_texture(&reset_before.texture);
    let output = fixture.base.read_texture(&reset_after.texture);
    let depth = fixture.read_depth(&reset_after);
    for (x, y) in [(SIZE.x / 2, SIZE.y / 2), (28, 26)] {
        assert!(pixel(&depth, x, y)[0] > 0.0, "interior depth at ({x},{y})");
        let stationary = motion_at(&motion, x, y);
        assert!(stationary.is_finite());
        assert!(
            stationary.x.abs() < 1.0e-3 && stationary.y.abs() < 1.0e-3,
            "stationary motion at ({x},{y}) = {stationary:?}"
        );
    }
    let center_motion = motion_at(&motion, SIZE.x / 2, SIZE.y / 2);
    fixture.last_pixels = format!(
        "reset_center={:?} depth_center={:?} motion_center={center_motion:?}",
        pixel(&source, SIZE.x / 2, SIZE.y / 2),
        pixel(&depth, SIZE.x / 2, SIZE.y / 2)
    );
    let center_source = pixel(&source, SIZE.x / 2, SIZE.y / 2);
    let center_output = pixel(&output, SIZE.x / 2, SIZE.y / 2);
    for channel in 0..3 {
        assert!(approx(center_source[channel], EXPECTED[channel], 8.0e-3));
        assert_eq!(center_source[3], 1.0);
        assert!(approx(
            center_output[channel],
            center_source[channel],
            8.0e-3
        ));
    }
    assert!(approx(
        pixel(&depth, SIZE.x / 2, SIZE.y / 2)[0],
        0.02,
        2.0e-3
    ));
    for (x, y) in [
        (0, 0),
        (SIZE.x - 1, 0),
        (0, SIZE.y - 1),
        (SIZE.x - 1, SIZE.y - 1),
    ] {
        assert_eq!(pixel(&depth, x, y)[0], 0.0, "corner depth");
    }
    let mut color_covered = 0usize;
    let mut depth_covered = 0usize;
    let mut mismatch = 0usize;
    for y in 0..SIZE.y {
        for x in 0..SIZE.x {
            let color = color_visible(pixel(&source, x, y));
            let d = pixel(&depth, x, y)[0] > 0.0;
            color_covered += usize::from(color);
            depth_covered += usize::from(d);
            mismatch += usize::from(color != d);
        }
    }
    assert!(color_covered > 0 && depth_covered > 0);
    assert!(
        mismatch <= 4,
        "color/depth coverage mismatch={mismatch}, color={color_covered}, depth={depth_covered}"
    );
    assert!(output.iter().copied().all(finite));
    for (source_pixel, output_pixel) in source.iter().zip(output.iter()) {
        for channel in 0..3 {
            assert!(
                approx(source_pixel[channel], output_pixel[channel], 8.0e-3),
                "reset passthrough mismatch: {source_pixel:?} vs {output_pixel:?}"
            );
        }
    }
    println!(
        "PASS A: terrain material color, native depth/motion, color={color_covered} depth={depth_covered} mismatch={mismatch}"
    );

    // B. Actual camera motion.
    fixture.set_camera_translation(0.1);
    fixture.update();
    let (motion_before, _) = fixture.pair();
    let motion_texture = motion_before
        .motion
        .as_ref()
        .expect("motion texture")
        .2
        .clone();
    let motion = fixture.base.read_texture(&motion_texture);
    let measured = motion_at(&motion, 30, 32);
    let expected = -0.1 / (2.0 * (30.0f32.to_radians()).tan() * 5.0);
    assert!(
        approx(measured.x, expected, 1.0e-3),
        "motion {measured:?} expected {expected}"
    );
    assert!(
        approx(measured.y, 0.0, 1.0e-3),
        "vertical motion {measured:?}"
    );
    let source = fixture.base.read_texture(&motion_before.texture);
    assert!(color_visible(pixel(&source, 30, 32)));
    let depth = fixture.read_depth(&motion_before);
    assert!(pixel(&depth, 30, 32)[0] > 0.0);
    fixture.last_pixels = format!(
        "camera_source={:?} depth={:?} motion={measured:?} expected_x={expected}",
        pixel(&source, 30, 32),
        pixel(&depth, 30, 32)
    );
    println!("PASS B: terrain camera motion={measured:?} expected_x={expected}");

    // C. Reconstruction, jitter and reset.
    fixture.set_camera_translation(0.0);
    fixture.base.request_reset();
    fixture.update();
    fixture.base.release_reset();
    fixture.wait("first terrain non-reset frame", |fixture| {
        if !fixture.sampling_ready() {
            return None;
        }
        let (before, after) = fixture.base.captures();
        let (Some(before), Some(after)) = (before, after) else {
            return None;
        };
        (!before.reset
            && !after.reset
            && before.taa_present
            && after.taa_present
            && before.view == after.view
            && before.frame == after.frame
            && before.texture.id() != after.texture.id()
            && before.jitter.is_some())
        .then_some(())
    });
    let mut jitters = Vec::new();
    let mut changed_edges = 0usize;
    for frame in 0..16 {
        fixture.update();
        let (before, after) = taa_executed_pair(&fixture);
        assert!(!before.reset && !after.reset, "frame {frame} reset");
        let source = fixture.base.read_texture(&before.texture);
        let output = fixture.base.read_texture(&after.texture);
        fixture.last_pixels = format!(
            "frame={frame} center={:?} output={:?}",
            pixel(&source, SIZE.x / 2, SIZE.y / 2),
            pixel(&output, SIZE.x / 2, SIZE.y / 2)
        );
        assert!(source.iter().copied().all(finite));
        assert!(output.iter().copied().all(finite));
        let jitter = before.jitter.expect("jitter");
        assert!(jitter.is_finite() && jitter.x.abs() <= 0.5 && jitter.y.abs() <= 0.5);
        jitters.push(jitter);
        for y in 0..SIZE.y {
            for x in 0..SIZE.x {
                let a = pixel(&source, x, y);
                let b = pixel(&output, x, y);
                let difference = (0..3)
                    .map(|channel| (a[channel] - b[channel]).abs())
                    .fold(0.0f32, f32::max);
                if edge_at(&source, x, y) && difference > 1.0e-3 {
                    changed_edges += 1;
                }
            }
        }
    }
    assert!(changed_edges > 0, "no reconstructed terrain edge");
    assert!(
        jitters.windows(2).any(|pair| pair[0] != pair[1]),
        "jitter did not vary"
    );
    fixture.set_camera_translation(0.15);
    fixture.base.request_reset();
    fixture.wait("terrain cut reset execution", |fixture| {
        if !fixture.sampling_ready() {
            return None;
        }
        let (before, after) = fixture.base.captures();
        let (Some(before), Some(after)) = (before, after) else {
            return None;
        };
        if !before.reset
            || !after.reset
            || !before.taa_present
            || !after.taa_present
            || before.view != after.view
            || before.frame != after.frame
            || before.texture.id() == after.texture.id()
        {
            return None;
        }
        let source = fixture.base.read_texture(&before.texture);
        let center = pixel(&source, SIZE.x / 2, SIZE.y / 2);
        let depth = fixture.read_depth(&after);
        fixture.last_pixels = format!(
            "cut_source_center={center:?} cut_depth_center={:?}",
            pixel(&depth, SIZE.x / 2, SIZE.y / 2)
        );
        (color_visible(center) && pixel(&depth, SIZE.x / 2, SIZE.y / 2)[0] > 0.0)
            .then_some((before, after))
    });
    let (reset_before, reset_after) = taa_executed_pair(&fixture);
    assert!(reset_before.reset && reset_after.reset);
    let reset_source = fixture.base.read_texture(&reset_before.texture);
    let reset_output = fixture.base.read_texture(&reset_after.texture);
    fixture.last_pixels = format!(
        "cut_source_center={:?} cut_output_center={:?}",
        pixel(&reset_source, SIZE.x / 2, SIZE.y / 2),
        pixel(&reset_output, SIZE.x / 2, SIZE.y / 2)
    );
    for (source_pixel, output_pixel) in reset_source.iter().zip(reset_output.iter()) {
        assert!(finite(*source_pixel) && finite(*output_pixel));
        for channel in 0..3 {
            assert!(approx(source_pixel[channel], output_pixel[channel], 8.0e-3));
        }
        assert_eq!(
            source_pixel[3], output_pixel[3],
            "reset must preserve scene alpha"
        );
    }
    println!(
        "PASS C: edge_count={changed_edges} jitter_varied={} reset passthrough",
        jitters.len()
    );

    // D. Hide/restore real coverage.
    fixture.base.request_reset();
    fixture
        .base
        .app
        .world_mut()
        .entity_mut(fixture.terrain)
        .insert(Visibility::Hidden);
    fixture.wait("hidden terrain background", |fixture| {
        if !fixture.sampling_ready() {
            return None;
        }
        let (before, _) = fixture.base.captures();
        let before = before?;
        let source = fixture.base.read_texture(&before.texture);
        let depth = fixture.read_depth(&before);
        let center = pixel(&source, SIZE.x / 2, SIZE.y / 2);
        let d = pixel(&depth, SIZE.x / 2, SIZE.y / 2)[0];
        fixture.last_pixels =
            format!("hidden_center={center:?} hidden_depth={d} expected={EXPECTED:?}");
        (center[0].max(center[1]).max(center[2]) <= 0.01 && d == 0.0).then_some(())
    });
    fixture
        .base
        .app
        .world_mut()
        .entity_mut(fixture.terrain)
        .insert(Visibility::Visible);
    fixture.base.request_reset();
    fixture.wait("restored terrain coverage", |fixture| {
        if !fixture.sampling_ready() {
            return None;
        }
        let (before, _) = fixture.base.captures();
        let before = before?;
        let source = fixture.base.read_texture(&before.texture);
        let depth = fixture.read_depth(&before);
        let center = pixel(&source, SIZE.x / 2, SIZE.y / 2);
        let d = pixel(&depth, SIZE.x / 2, SIZE.y / 2)[0];
        fixture.last_pixels = format!("restored_center={center:?} restored_depth={d}");
        (color_visible(center) && d > 0.0).then_some(())
    });
    println!("PASS D: hide produced background depth 0 and restore recovered real coverage");

    // E. Missing attachment controls.
    let camera = fixture.base.camera;
    fixture
        .base
        .app
        .world_mut()
        .entity_mut(camera)
        .remove::<MotionVectorPrepass>();
    fixture.wait("missing motion attachment", |fixture| {
        if !fixture.sampling_ready() {
            return None;
        }
        let (before, after) = fixture.base.captures();
        let (Some(before), Some(after)) = (before, after) else {
            return None;
        };
        if before.view != after.view || before.frame != after.frame {
            return None;
        }
        let source = fixture.base.read_texture(&before.texture);
        let center = pixel(&source, SIZE.x / 2, SIZE.y / 2);
        fixture.last_pixels = format!(
            "missing_motion_center={center:?} expected={EXPECTED:?} ids_equal={}",
            before.texture.id() == after.texture.id()
        );
        (before.taa_present
            && after.taa_present
            && before.motion.is_none()
            && before.texture.id() == after.texture.id()
            && finite(center)
            && color_visible(center)
            && (0..3).all(|channel| approx(center[channel], EXPECTED[channel], 8.0e-3)))
        .then_some(())
    });
    fixture
        .base
        .app
        .world_mut()
        .entity_mut(camera)
        .insert(MotionVectorPrepass);
    fixture.base.request_reset();
    let (motion_reset_before, motion_reset_after) =
        fixture.wait("motion TAA restoration", |fixture| {
            if !fixture.sampling_ready() {
                return None;
            }
            let (before, after) = fixture.base.captures();
            let (Some(before), Some(after)) = (before, after) else {
                return None;
            };
            let motion = before.motion.as_ref()?;
            if motion.0 != MOTION_VECTOR_PREPASS_FORMAT
                || motion.1.width != SIZE.x
                || motion.1.height != SIZE.y
                || !before.reset
                || !after.reset
                || !before.taa_present
                || before.view != after.view
                || before.frame != after.frame
                || before.texture.id() == after.texture.id()
            {
                return None;
            }
            let source = fixture.base.read_texture(&before.texture);
            let depth = fixture.read_depth(&before);
            let center = pixel(&source, SIZE.x / 2, SIZE.y / 2);
            fixture.last_pixels = format!(
                "restored_motion_center={center:?} depth={:?}",
                pixel(&depth, SIZE.x / 2, SIZE.y / 2)
            );
            (color_visible(center) && pixel(&depth, SIZE.x / 2, SIZE.y / 2)[0] > 0.0)
                .then_some((before, after))
        });
    let motion_source = fixture.base.read_texture(&motion_reset_before.texture);
    let motion_output = fixture.base.read_texture(&motion_reset_after.texture);
    for (source_pixel, output_pixel) in motion_source.iter().zip(motion_output.iter()) {
        assert!(finite(*source_pixel) && finite(*output_pixel));
        for channel in 0..3 {
            assert!(approx(source_pixel[channel], output_pixel[channel], 8.0e-3));
        }
        assert_eq!(source_pixel[3], output_pixel[3]);
    }

    fixture
        .base
        .app
        .world_mut()
        .entity_mut(camera)
        .remove::<DepthPrepass>();
    fixture.wait("missing depth attachment", |fixture| {
        if !fixture.sampling_ready() {
            return None;
        }
        let (before, after) = fixture.base.captures();
        let (Some(before), Some(after)) = (before, after) else {
            return None;
        };
        if before.view != after.view || before.frame != after.frame {
            return None;
        }
        let source = fixture.base.read_texture(&before.texture);
        let center = pixel(&source, SIZE.x / 2, SIZE.y / 2);
        fixture.last_pixels = format!(
            "missing_depth_center={center:?} expected={EXPECTED:?} ids_equal={}",
            before.texture.id() == after.texture.id()
        );
        (before.taa_present
            && after.taa_present
            && before.depth.is_none()
            && before.texture.id() == after.texture.id()
            && finite(center)
            && color_visible(center)
            && (0..3).all(|channel| approx(center[channel], EXPECTED[channel], 8.0e-3)))
        .then_some(())
    });
    fixture
        .base
        .app
        .world_mut()
        .entity_mut(camera)
        .insert(DepthPrepass);
    fixture.base.request_reset();
    let (depth_reset_before, depth_reset_after) =
        fixture.wait("depth TAA restoration", |fixture| {
            if !fixture.sampling_ready() {
                return None;
            }
            let (before, after) = fixture.base.captures();
            let (Some(before), Some(after)) = (before, after) else {
                return None;
            };
            let depth = before.depth.as_ref()?;
            if depth.0 != TextureFormat::Depth32Float
                || depth.1.width != SIZE.x
                || depth.1.height != SIZE.y
                || !before.reset
                || !after.reset
                || !before.taa_present
                || before.view != after.view
                || before.frame != after.frame
                || before.texture.id() == after.texture.id()
            {
                return None;
            }
            let source = fixture.base.read_texture(&before.texture);
            let depth_pixels = fixture.read_depth(&before);
            let center = pixel(&source, SIZE.x / 2, SIZE.y / 2);
            fixture.last_pixels = format!(
                "restored_depth_center={center:?} depth={:?}",
                pixel(&depth_pixels, SIZE.x / 2, SIZE.y / 2)
            );
            (color_visible(center) && pixel(&depth_pixels, SIZE.x / 2, SIZE.y / 2)[0] > 0.0)
                .then_some((before, after))
        });
    let depth_source = fixture.base.read_texture(&depth_reset_before.texture);
    let depth_output = fixture.base.read_texture(&depth_reset_after.texture);
    for (source_pixel, output_pixel) in depth_source.iter().zip(depth_output.iter()) {
        assert!(finite(*source_pixel) && finite(*output_pixel));
        for channel in 0..3 {
            assert!(approx(source_pixel[channel], output_pixel[channel], 8.0e-3));
        }
        assert_eq!(source_pixel[3], output_pixel[3]);
    }
    println!("PASS E: missing motion/depth suppress TAA; restored attachments resume it");
}
