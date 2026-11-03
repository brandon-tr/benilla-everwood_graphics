use super::*;

use benilla_assets::coords::wow_to_bevy;
use bevy::anti_alias::taa::TemporalAntiAliasing;
use bevy::core_pipeline::prepass::{
    DepthPrepass, MotionVectorPrepass, NormalPrepass, ViewPrepassTextures,
    MOTION_VECTOR_PREPASS_FORMAT,
};
use bevy::render::render_resource::TextureFormat;

use super::super::TerrainReferenceFixture;
use crate::aa::bevy_reference::ObservedCapture;
use crate::static_gx::RetainedTemporalSource;

const SIZE: UVec2 = super::super::super::SIZE;
const OPAQUE: [f32; 4] = [1.0, 96.0 / 255.0, 32.0 / 255.0, 1.0];
const CUTOUT: [f32; 4] = [1.0, 0.0, 1.0, 1.0];
const TERRAIN: [f32; 4] = [0.2, 0.6, 0.902, 1.0];

fn pixel(pixels: &[[f32; 4]], x: u32, y: u32) -> [f32; 4] {
    pixels[(y * SIZE.x + x) as usize]
}

fn finite(p: [f32; 4]) -> bool {
    p.iter().all(|v| v.is_finite())
}

fn approx(a: f32, b: f32, tolerance: f32) -> bool {
    (a - b).abs() <= tolerance
}

fn visible(p: [f32; 4]) -> bool {
    p[0].max(p[1]).max(p[2]) > 0.05
}

fn color_is(actual: [f32; 4], expected: [f32; 4], tolerance: f32) -> bool {
    actual[3] >= 0.99 && (0..3).all(|i| approx(actual[i], expected[i], tolerance))
}

fn decoded_normal(p: [f32; 4]) -> Vec3 {
    Vec3::new(p[0] * 2.0 - 1.0, p[1] * 2.0 - 1.0, p[2] * 2.0 - 1.0)
}

fn normal_is(p: [f32; 4], expected: Vec3) -> bool {
    p[3] >= 0.99 && decoded_normal(p).distance(expected) <= 1.0e-2
}

fn depth_far(z: f32) -> f32 {
    // Isolated camera is 5 units from z=0, default perspective near=0.1.
    0.1 / (5.0 - z)
}

fn native_inputs(terrain: &TerrainReferenceFixture, capture: &ObservedCapture) -> bool {
    let Some(prepass) = terrain
        .base
        .render_world()
        .get::<ViewPrepassTextures>(capture.view)
    else {
        return false;
    };
    let (Some(depth), Some(motion)) = (prepass.depth.as_ref(), prepass.motion_vectors.as_ref())
    else {
        return false;
    };
    let depth = &depth.texture.texture;
    let motion = &motion.texture.texture;
    depth.format() == TextureFormat::Depth32Float
        && depth.sample_count() == 1
        && depth.width() == SIZE.x
        && depth.height() == SIZE.y
        && motion.format() == MOTION_VECTOR_PREPASS_FORMAT
        && motion.sample_count() == 1
        && motion.width() == SIZE.x
        && motion.height() == SIZE.y
}

fn reset_executed(terrain: &TerrainReferenceFixture, snap: &Snap) -> bool {
    snap.before.view == snap.after.view
        && snap.before.frame == snap.after.frame
        && snap.before.taa_present
        && snap.after.taa_present
        && snap.before.reset
        && snap.after.reset
        && snap.before.texture.id() != snap.after.texture.id()
        && native_inputs(terrain, &snap.before)
}

struct Snap {
    before: ObservedCapture,
    after: ObservedCapture,
    source: Vec<[f32; 4]>,
    depth: Option<Vec<[f32; 4]>>,
    motion: Option<Vec<[f32; 4]>>,
    normal: Option<Vec<[f32; 4]>>,
}

fn snapshot(terrain: &mut TerrainReferenceFixture) -> Option<Snap> {
    if !terrain.sampling_ready() {
        return None;
    }
    let (before, after) = terrain.base.captures();
    let (Some(before), Some(after)) = (before, after) else {
        return None;
    };
    if before.view != after.view || before.frame != after.frame {
        return None;
    }
    let source = terrain.base.read_texture(&before.texture);
    let depth = before.depth.is_some().then(|| terrain.read_depth(&after));
    let motion = before.motion.as_ref().map(|m| {
        let texture = m.2.clone();
        terrain.base.read_texture(&texture)
    });
    let normal = {
        let render_world = terrain.base.render_world();
        let texture = render_world
            .get::<ViewPrepassTextures>(before.view)
            .and_then(|prepass| prepass.normal.as_ref())
            .map(|normal| normal.texture.texture.clone());
        texture.map(|texture| terrain.base.read_texture(&texture))
    };
    let diagnostics = {
        let render_world = terrain.base.render_world();
        render_world
            .resource::<RetainedTemporalSource>()
            .native_draw_diagnostics(render_world, before.view)
    };
    terrain.last_pixels = format!(
        "center={:?} depth={:?} normal={:?} motion={:?} reset={} taa={} frame={} diag=[{diagnostics}]",
        pixel(&source, SIZE.x / 2, SIZE.y / 2),
        depth.as_ref().map(|d| pixel(d, SIZE.x / 2, SIZE.y / 2)),
        normal.as_ref().map(|n| pixel(n, SIZE.x / 2, SIZE.y / 2)),
        motion.as_ref().map(|m| pixel(m, SIZE.x / 2, SIZE.y / 2)),
        before.reset,
        before.taa_present,
        before.frame
    );
    Some(Snap {
        before,
        after,
        source,
        depth,
        motion,
        normal,
    })
}

fn fail<T>(terrain: &mut TerrainReferenceFixture, predicate: &str) -> Option<T> {
    terrain.last_pixels = format!("{} predicate={predicate}", terrain.last_pixels);
    None
}

fn wait_visible(fixture: &mut RetainedReferenceFixture, stage: &str, expected: [f32; 4]) -> Snap {
    fixture.wait(stage, |terrain| {
        let snap = snapshot(terrain)?;
        let center = pixel(&snap.source, SIZE.x / 2, SIZE.y / 2);
        let depth = snap
            .depth
            .as_ref()
            .map(|d| pixel(d, SIZE.x / 2, SIZE.y / 2)[0])
            .unwrap_or(0.0);
        if color_is(center, expected, 0.025) && depth > 0.0 {
            Some(snap)
        } else {
            fail(
                terrain,
                "expected visible retained color and positive center depth",
            )
        }
    })
}

fn wait_restored(
    fixture: &mut RetainedReferenceFixture,
    stage: &str,
    expected: [f32; 4],
    plane_z: f32,
) -> Snap {
    fixture.wait(stage, |terrain| {
        let snap = snapshot(terrain)?;
        let center = pixel(&snap.source, SIZE.x / 2, SIZE.y / 2);
        let depth = snap
            .depth
            .as_ref()
            .map(|d| pixel(d, SIZE.x / 2, SIZE.y / 2)[0])
            .unwrap_or(0.0);
        if reset_executed(terrain, &snap)
            && color_is(center, expected, 0.025)
            && approx(depth, depth_far(plane_z), 1.5e-3)
        {
            Some(snap)
        } else {
            fail(
                terrain,
                "expected reset execution, native inputs, visible color and analytic depth",
            )
        }
    })
}

fn wait_stationary(
    fixture: &mut RetainedReferenceFixture,
    stage: &str,
    expected: [f32; 4],
    plane_z: f32,
) -> Snap {
    fixture.wait(stage, |terrain| {
        let snap = snapshot(terrain)?;
        let center = pixel(&snap.source, SIZE.x / 2, SIZE.y / 2);
        let depth = snap
            .depth
            .as_ref()
            .map(|d| pixel(d, SIZE.x / 2, SIZE.y / 2)[0])
            .unwrap_or(0.0);
        let Some(motion) = snap.motion.as_ref() else {
            return fail::<Snap>(terrain, "stationary requires a motion attachment");
        };
        let stationary = [(SIZE.x / 2, SIZE.y / 2), (26, 28)].iter().all(|(x, y)| {
            let m = pixel(motion, *x, *y);
            finite(m) && m[0].abs() < 1.0e-3 && m[1].abs() < 1.0e-3
        });
        if color_is(center, expected, 0.025)
            && approx(depth, depth_far(plane_z), 1.5e-3)
            && snap.before.taa_present
            && snap.after.taa_present
            && snap.before.texture.id() != snap.after.texture.id()
            && native_inputs(terrain, &snap.before)
            && stationary
        {
            Some(snap)
        } else {
            fail(
                terrain,
                "expected settled stationary motion, TAA execution, color and analytic depth",
            )
        }
    })
}

fn wait_normal(
    fixture: &mut RetainedReferenceFixture,
    stage: &str,
    expected: [f32; 4],
    plane_z: f32,
    normal: Vec3,
) -> Snap {
    fixture.wait(stage, |terrain| {
        let snap = snapshot(terrain)?;
        let center = pixel(&snap.source, SIZE.x / 2, SIZE.y / 2);
        let depth = pixel(snap.depth.as_ref()?, SIZE.x / 2, SIZE.y / 2)[0];
        let encoded = pixel(snap.normal.as_ref()?, SIZE.x / 2, SIZE.y / 2);
        (reset_executed(terrain, &snap)
            && color_is(center, expected, 0.025)
            && approx(depth, depth_far(plane_z), 1.5e-3)
            && normal_is(encoded, normal))
        .then_some(snap)
        .or_else(|| {
            fail(
                terrain,
                "expected reset TAA, color, depth and encoded world normal",
            )
        })
    })
}

fn normal_input(terrain: &RetainedReferenceFixture, capture: &ObservedCapture) -> bool {
    let Some(prepass) = terrain
        .terrain
        .base
        .render_world()
        .get::<ViewPrepassTextures>(capture.view)
    else {
        return false;
    };
    let Some(normal) = prepass.normal.as_ref() else {
        return false;
    };
    let normal = &normal.texture.texture;
    normal.format() == bevy::core_pipeline::prepass::NORMAL_PREPASS_FORMAT
        && normal.sample_count() == 1
        && normal.width() == SIZE.x
        && normal.height() == SIZE.y
}

fn wait_invisible(fixture: &mut RetainedReferenceFixture, stage: &str) -> Snap {
    fixture.wait(stage, |terrain| {
        let snap = snapshot(terrain)?;
        let color_any = snap
            .source
            .iter()
            .copied()
            .any(|p| visible(p) && p[3] >= 0.99);
        let depth_any = snap
            .depth
            .as_ref()
            .is_some_and(|d| d.iter().any(|p| p[0] > 0.0));
        if !color_any && !depth_any {
            Some(snap)
        } else {
            fail(
                terrain,
                "expected no retained color and no retained prepass depth",
            )
        }
    })
}

#[test]
fn retained_submesh_faces_camera_after_coordinate_conversion() {
    let mesh = retained_submesh(ModelBlend::Opaque);
    let bevy_positions = mesh
        .positions
        .iter()
        .map(|p| wow_to_bevy(*p))
        .collect::<Vec<_>>();
    assert_eq!(bevy_positions.len(), 4);
    for p in &bevy_positions {
        assert!(approx(p.z, 0.0, 1.0e-6), "constant local Z: {p:?}");
    }
    assert!(approx(bevy_positions[0].x, -1.5, 1.0e-6));
    assert!(approx(bevy_positions[2].x, 1.5, 1.0e-6));
    assert!(approx(bevy_positions[0].y, -1.5, 1.0e-6));
    assert!(approx(bevy_positions[2].y, 1.5, 1.0e-6));
    let normals = mesh
        .normals
        .iter()
        .map(|n| wow_to_bevy(*n))
        .collect::<Vec<_>>();
    for n in &normals {
        assert!(approx(n.x, 0.0, 1.0e-6) && approx(n.y, 0.0, 1.0e-6));
        assert!(approx(n.z, 1.0, 1.0e-6), "normal must face Bevy +Z: {n:?}");
    }
    let a = bevy_positions[0];
    let b = bevy_positions[1];
    let c = bevy_positions[2];
    let cross = (b - a).cross(c - a);
    assert!(cross.z > 0.0, "triangles must wind toward +Z: {cross:?}");
}

#[test]
#[ignore = "Requires a GPU; run explicitly on the host PC"]
fn gpu_retained_native_taa_readback() {
    let mut fixture = RetainedReferenceFixture::new();
    fixture.terrain.print_adapter();
    fixture.hide_terrain(true);

    // A. Opaque retained-only surface.
    fixture.terrain.base.request_reset();
    let warm = wait_restored(&mut fixture, "opaque retained warmup", OPAQUE, 0.05);
    assert!(native_inputs(&fixture.terrain, &warm.before));
    assert_eq!(
        warm.before.depth.expect("depth").0,
        TextureFormat::Depth32Float
    );
    assert_eq!(
        warm.before.motion.expect("motion").0,
        MOTION_VECTOR_PREPASS_FORMAT
    );
    let depth = warm.depth.as_ref().expect("depth pixels");
    let motion = warm.motion.as_ref().expect("motion pixels");
    for (x, y) in [(SIZE.x / 2, SIZE.y / 2), (26, 28)] {
        assert!(pixel(&warm.source, x, y)[3] >= 0.99);
        assert!(color_is(pixel(&warm.source, x, y), OPAQUE, 0.025));
        assert!(pixel(depth, x, y)[0] > 0.0);
        assert!(approx(pixel(depth, x, y)[0], depth_far(0.05), 1.5e-3));
        let m = pixel(motion, x, y);
        assert!(finite(m));
        assert!(m[0].abs() < 1.0e-3 && m[1].abs() < 1.0e-3);
    }
    for (x, y) in [
        (0, 0),
        (SIZE.x - 1, 0),
        (0, SIZE.y - 1),
        (SIZE.x - 1, SIZE.y - 1),
    ] {
        assert_eq!(pixel(depth, x, y)[0], 0.0, "corner background depth");
    }
    let (before, after) = fixture.pair();
    assert!(before.taa_present && after.taa_present);
    assert_ne!(before.texture.id(), after.texture.id());
    println!("PASS A: opaque retained source/depth/motion and TAA flip");

    // B. One camera translation, measured against the retained plane.
    fixture.terrain.set_camera_translation(0.1);
    fixture.update();
    let (before, after) = fixture.pair();
    assert!(before.taa_present && after.taa_present);
    assert_ne!(before.texture.id(), after.texture.id());
    let source = fixture.terrain.base.read_texture(&before.texture);
    let depth = fixture.read_depth(&before);
    let motion_texture = before.motion.as_ref().expect("motion").2.clone();
    let motion = fixture.terrain.base.read_texture(&motion_texture);
    let measured = pixel(&motion, SIZE.x / 2, SIZE.y / 2);
    let expected = -0.1 / (2.0 * (30.0f32.to_radians()).tan() * 4.95);
    assert!(
        approx(measured[0], expected, 1.0e-3),
        "{measured:?} vs {expected}"
    );
    assert!(approx(measured[1], 0.0, 1.0e-3));
    assert!(color_is(
        pixel(&source, SIZE.x / 2, SIZE.y / 2),
        OPAQUE,
        0.025
    ));
    assert!(approx(
        pixel(&depth, SIZE.x / 2, SIZE.y / 2)[0],
        depth_far(0.05),
        1.5e-3
    ));
    println!("PASS B: camera motion={measured:?} expected={expected}");

    // C. Cutout hole over terrain moved clearly farther back.
    fixture.terrain.set_camera_translation(0.0);
    fixture.set_terrain_depth(-4.0);
    fixture.hide_terrain(false);
    fixture.set_cutout(Vec3::new(0.0, 0.0, 0.3));
    fixture.terrain.base.request_reset();
    let cut = fixture.wait("cutout over terrain", |terrain| {
        let snap = snapshot(terrain)?;
        let center = pixel(&snap.source, SIZE.x / 2, SIZE.y / 2);
        let opaque = pixel(&snap.source, 24, 24);
        let depth = snap.depth.as_ref()?;
        let center_depth = pixel(depth, SIZE.x / 2, SIZE.y / 2)[0];
        let opaque_depth = pixel(depth, 24, 24)[0];
        if color_is(center, TERRAIN, 0.03)
            && color_is(opaque, CUTOUT, 0.03)
            && approx(center_depth, depth_far(-4.0), 1.0e-3)
            && approx(opaque_depth, depth_far(0.3), 1.5e-3)
        {
            Some(snap)
        } else {
            fail(
                terrain,
                "cutout: terrain through hole and opaque magenta edge",
            )
        }
    });
    assert!(cut.after.taa_present);
    println!("PASS C: cutout hole and opaque retained card over separated terrain");

    // D. Back-facing one-sided culling, then the two-sided variant.
    fixture.set_terrain_depth(0.0);
    fixture.hide_terrain(true);
    fixture.set_opaque(Vec3::new(0.0, 0.0, 0.3), false, std::f32::consts::PI);
    fixture.terrain.base.request_reset();
    let culled = wait_invisible(&mut fixture, "back-facing one-sided");
    assert!(culled
        .depth
        .as_ref()
        .is_some_and(|d| d.iter().all(|p| p[0] == 0.0)));
    fixture.set_opaque(Vec3::new(0.0, 0.0, 0.3), true, std::f32::consts::PI);
    fixture.terrain.base.request_reset();
    let two_sided = wait_visible(&mut fixture, "two-sided back face", OPAQUE);
    assert!(two_sided.depth.as_ref().unwrap()[pixel_index(32, 32)][0] > 0.0);
    println!("PASS D: one-sided absent, two-sided present");

    // E. Both overlap directions with separated planes.
    fixture.hide_terrain(false);
    fixture.set_terrain_depth(-2.0);
    fixture.set_opaque(Vec3::new(0.0, 0.0, 0.3), true, 0.0);
    fixture.terrain.base.request_reset();
    let _ = wait_visible(&mut fixture, "retained in front first visible", OPAQUE);
    let retained_front = wait_stationary(
        &mut fixture,
        "retained in front settled stationary",
        OPAQUE,
        0.3,
    );
    let rf_depth = retained_front.depth.as_ref().unwrap();
    assert!(approx(pixel(rf_depth, 32, 32)[0], depth_far(0.3), 1.5e-3));
    let rf_motion = retained_front
        .motion
        .as_ref()
        .expect("retained-front motion attachment");
    for (x, y) in [(32, 32), (26, 28)] {
        let m = pixel(rf_motion, x, y);
        assert!(m[0].abs() < 1.0e-3 && m[1].abs() < 1.0e-3);
    }
    fixture.set_terrain_depth(1.0);
    fixture.set_opaque(Vec3::new(0.0, 0.0, -1.0), true, 0.0);
    fixture.terrain.base.request_reset();
    let _ = fixture.wait("terrain in front first visible", |terrain| {
        let snap = snapshot(terrain)?;
        let c = pixel(&snap.source, SIZE.x / 2, SIZE.y / 2);
        let d = pixel(snap.depth.as_ref()?, SIZE.x / 2, SIZE.y / 2)[0];
        (color_is(c, TERRAIN, 0.03) && approx(d, depth_far(1.0), 1.2e-3))
            .then_some(snap)
            .or_else(|| fail(terrain, "terrain in front must win color/depth"))
    });
    let terrain_front = wait_stationary(
        &mut fixture,
        "terrain in front settled stationary",
        TERRAIN,
        1.0,
    );
    assert!(terrain_front.depth.as_ref().unwrap()[pixel_index(32, 32)][0] > 0.0);
    let tf_motion = terrain_front
        .motion
        .as_ref()
        .expect("terrain-front motion attachment");
    for (x, y) in [(32, 32), (26, 28)] {
        let m = pixel(tf_motion, x, y);
        assert!(m[0].abs() < 1.0e-3 && m[1].abs() < 1.0e-3);
    }
    println!("PASS E: retained-front and terrain-front overlap, settled stationary");

    // F. Removal covers the whole image; then restore real geometry.
    fixture.set_terrain_depth(0.0);
    fixture.hide_terrain(true);
    fixture.set_opaque(Vec3::new(0.0, 0.0, 0.05), true, 0.0);
    fixture.terrain.base.request_reset();
    let _ = wait_visible(&mut fixture, "pre-removal coverage", OPAQUE);
    fixture.release_retained();
    fixture.clear_retained();
    fixture.terrain.base.request_reset();
    let removed = wait_invisible(&mut fixture, "after owner release");
    assert!(removed.source.iter().all(|p| !visible(*p)));
    fixture.set_opaque(Vec3::new(0.0, 0.0, 0.05), true, 0.0);
    fixture.terrain.base.request_reset();
    let restored = wait_visible(&mut fixture, "restored opaque", OPAQUE);
    assert!(restored.depth.unwrap()[pixel_index(32, 32)][0] > 0.0);
    println!("PASS F: full-image removal and real re-divert");

    // G. Camera cut/reset must use captures from the new camera pose.
    fixture.terrain.set_camera_translation(0.15);
    fixture.update();
    fixture.terrain.base.request_reset();
    let reset = wait_restored(&mut fixture, "reset after camera cut", OPAQUE, 0.05);
    assert!(reset.before.reset && reset.after.reset);
    assert!(reset.before.taa_present && reset.after.taa_present);
    let reset_before = reset.before;
    let reset_after = reset.after;
    assert_eq!(reset_before.frame, reset_after.frame);
    let rs = reset.source;
    let ro = fixture.terrain.base.read_texture(&reset_after.texture);
    assert!(visible(pixel(&rs, SIZE.x / 2, SIZE.y / 2)));
    for (s, o) in rs.iter().zip(ro.iter()) {
        assert!(finite(*s) && finite(*o));
        for channel in 0..3 {
            assert!(approx(s[channel], o[channel], 8.0e-3));
        }
        assert_eq!(s[3], o[3]);
    }
    println!("PASS G: reset from the new camera pose matches source");

    // H. Missing native inputs, then restoration.
    let camera = fixture.terrain.base.camera;
    fixture
        .terrain
        .base
        .app
        .world_mut()
        .entity_mut(camera)
        .remove::<MotionVectorPrepass>();
    fixture.terrain.base.request_reset();
    let missing_motion = fixture.wait("missing motion", |terrain| {
        let snap = snapshot(terrain)?;
        (snap.before.motion.is_none()
            && snap.before.taa_present
            && snap.after.taa_present
            && snap.before.texture.id() == snap.after.texture.id()
            && color_is(pixel(&snap.source, SIZE.x / 2, SIZE.y / 2), OPAQUE, 0.025))
        .then_some(snap)
        .or_else(|| {
            fail(
                terrain,
                "motion absent, raw color visible, TAA present, no flip",
            )
        })
    });
    assert!(missing_motion.depth.is_some());
    fixture
        .terrain
        .base
        .app
        .world_mut()
        .entity_mut(camera)
        .insert(MotionVectorPrepass);
    fixture.terrain.base.request_reset();
    let restored = wait_restored(&mut fixture, "motion restored", OPAQUE, 0.05);
    assert!(restored.before.motion.is_some());
    fixture
        .terrain
        .base
        .app
        .world_mut()
        .entity_mut(camera)
        .remove::<DepthPrepass>();
    fixture.terrain.base.request_reset();
    let missing_depth = fixture.wait("missing depth", |terrain| {
        let snap = snapshot(terrain)?;
        (snap.before.depth.is_none()
            && snap.before.taa_present
            && snap.after.taa_present
            && snap.before.texture.id() == snap.after.texture.id()
            && color_is(pixel(&snap.source, SIZE.x / 2, SIZE.y / 2), OPAQUE, 0.025))
        .then_some(snap)
        .or_else(|| {
            fail(
                terrain,
                "depth absent, raw color visible, TAA present, no flip",
            )
        })
    });
    assert!(missing_depth.depth.is_none());
    fixture
        .terrain
        .base
        .app
        .world_mut()
        .entity_mut(camera)
        .insert(DepthPrepass);
    fixture.terrain.base.request_reset();
    let _ = wait_restored(&mut fixture, "depth restored", OPAQUE, 0.05);
    println!("PASS H: missing motion/depth suppress and restore");

    // I. NormalPrepass now carries retained world-space normals.
    fixture.terrain.base.request_reset();
    fixture
        .terrain
        .base
        .app
        .world_mut()
        .entity_mut(camera)
        .insert(NormalPrepass);
    let normal = fixture.wait("normal prepass retained output", |terrain| {
        let snap = snapshot(terrain)?;
        let center = pixel(&snap.source, SIZE.x / 2, SIZE.y / 2);
        let depth = pixel(snap.depth.as_ref()?, SIZE.x / 2, SIZE.y / 2)[0];
        let normal = pixel(snap.normal.as_ref()?, SIZE.x / 2, SIZE.y / 2);
        (color_is(center, OPAQUE, 0.025)
            && approx(depth, depth_far(0.05), 1.5e-3)
            && normal_is(normal, Vec3::Z)
            && reset_executed(terrain, &snap))
        .then_some(snap)
        .or_else(|| {
            fail(
                terrain,
                "NormalPrepass: retained color/depth/normal and reset TAA",
            )
        })
    });
    assert!(normal.normal.is_some());
    fixture
        .terrain
        .base
        .app
        .world_mut()
        .entity_mut(camera)
        .remove::<NormalPrepass>();
    fixture.terrain.base.request_reset();
    let _ = wait_restored(&mut fixture, "normal guard recovery", OPAQUE, 0.05);
    println!("PASS I: NormalPrepass carries retained +Z normal and recovers");

    // J. Native TAA removal leaves raw color and native prepass, without this adapter's depth.
    fixture
        .terrain
        .base
        .app
        .world_mut()
        .entity_mut(camera)
        .remove::<TemporalAntiAliasing>();
    fixture.terrain.base.request_reset();
    let no_taa = fixture.wait("native TAA removed", |terrain| {
        let snap = snapshot(terrain)?;
        (!snap.before.taa_present
            && snap.before.texture.id() == snap.after.texture.id()
            && color_is(pixel(&snap.source, SIZE.x / 2, SIZE.y / 2), OPAQUE, 0.025)
            && snap
                .depth
                .as_ref()
                .is_some_and(|d| d.iter().all(|p| p[0] == 0.0)))
        .then_some(snap)
        .or_else(|| {
            fail(
                terrain,
                "TAA removed: raw color, no flip, no retained depth",
            )
        })
    });
    assert!(!no_taa.before.taa_present);
    println!("PASS J: native TAA removal leaves raw retained color without adapter depth");
}

#[test]
#[ignore = "Requires a GPU; run explicitly on the host PC"]
fn gpu_retained_native_normal_readback() {
    let mut fixture = RetainedReferenceFixture::new();
    fixture.terrain.print_adapter();
    fixture.hide_terrain(true);
    let camera = fixture.terrain.base.camera;
    fixture
        .terrain
        .base
        .app
        .world_mut()
        .entity_mut(camera)
        .insert(NormalPrepass);

    fixture.set_opaque(Vec3::new(0.0, 0.0, 0.05), true, 0.0);
    fixture.terrain.base.request_reset();
    let plus_z = wait_normal(&mut fixture, "normal +Z", OPAQUE, 0.05, Vec3::Z);
    assert!(normal_input(&fixture, &plus_z.before));

    fixture.wait("normal stationary motion", |terrain| {
        let snap = snapshot(terrain)?;
        let motion = snap.motion.as_ref()?;
        let normal = snap.normal.as_ref()?;
        let depth = snap.depth.as_ref()?;
        let ok = [(SIZE.x / 2, SIZE.y / 2), (26, 28)].iter().all(|(x, y)| {
            let m = pixel(motion, *x, *y);
            color_is(pixel(&snap.source, *x, *y), OPAQUE, 0.025)
                && pixel(depth, *x, *y)[0] > 0.0
                && normal_is(pixel(normal, *x, *y), Vec3::Z)
                && finite(m)
                && m[0].abs() < 1.0e-3
                && m[1].abs() < 1.0e-3
        });
        (ok && reset_executed(terrain, &snap))
            .then_some(())
            .or_else(|| fail(terrain, "normal stationary: color/depth/normal/motion/TAA"))
    });

    let yaw = std::f32::consts::FRAC_PI_4;
    fixture.set_opaque(Vec3::new(0.0, 0.0, 0.05), true, yaw);
    fixture.terrain.base.request_reset();
    let rotated = Quat::from_rotation_y(yaw) * Vec3::Z;
    let _ = wait_normal(&mut fixture, "normal rotated", OPAQUE, 0.05, rotated);

    fixture.set_opaque(Vec3::new(0.0, 0.0, 0.05), true, std::f32::consts::PI);
    fixture.terrain.base.request_reset();
    let _ = wait_normal(
        &mut fixture,
        "normal back-facing two-sided",
        OPAQUE,
        0.05,
        -Vec3::Z,
    );

    // D. Native-terrain normal reference at the same pose, before the cutout card.
    fixture.set_terrain_depth(-4.0);
    fixture.hide_terrain(false);
    fixture.clear_retained();
    fixture.terrain.base.request_reset();
    let terrain_ref = fixture.wait("terrain normal reference", |terrain| {
        let snap = snapshot(terrain)?;
        let center = pixel(&snap.source, SIZE.x / 2, SIZE.y / 2);
        let depth = pixel(snap.depth.as_ref()?, SIZE.x / 2, SIZE.y / 2)[0];
        let normal = pixel(snap.normal.as_ref()?, SIZE.x / 2, SIZE.y / 2);
        (color_is(center, TERRAIN, 0.03)
            && approx(depth, depth_far(-4.0), 1.2e-3)
            && normal[3] >= 0.99)
            .then_some(snap)
            .or_else(|| fail(terrain, "terrain-only normal reference"))
    });
    let terrain_normal = pixel(
        terrain_ref.normal.as_ref().expect("terrain normal"),
        SIZE.x / 2,
        SIZE.y / 2,
    );

    // Cutout hole shows native terrain's normal; the opaque corner keeps the rotated retained normal.
    fixture.set_cutout_yaw(Vec3::new(0.0, 0.0, 0.3), yaw);
    fixture.terrain.base.request_reset();
    let cut = fixture.wait("normal cutout/terrain", |terrain| {
        let snap = snapshot(terrain)?;
        let center = pixel(&snap.source, SIZE.x / 2, SIZE.y / 2);
        // Same screen-X column as centre: outside the alpha hole but without the yawed plane's
        // large off-axis depth change.
        let opaque = pixel(&snap.source, 32, 24);
        let center_n = pixel(snap.normal.as_ref()?, SIZE.x / 2, SIZE.y / 2);
        let opaque_n = pixel(snap.normal.as_ref()?, 32, 24);
        let depth = snap.depth.as_ref()?;
        (color_is(center, TERRAIN, 0.03)
            && color_is(opaque, CUTOUT, 0.03)
            && normal_is(center_n, decoded_normal(terrain_normal))
            && normal_is(opaque_n, rotated)
            && approx(
                pixel(depth, SIZE.x / 2, SIZE.y / 2)[0],
                depth_far(-4.0),
                1.2e-3,
            )
            && approx(pixel(depth, 32, 24)[0], depth_far(0.3), 1.5e-3)
            && reset_executed(terrain, &snap))
        .then_some(snap)
        .or_else(|| fail(terrain, "cutout normal coverage"))
    });
    assert!(cut.normal.is_some());

    // Both overlap directions.
    fixture.set_terrain_depth(-2.0);
    fixture.set_opaque(Vec3::new(0.0, 0.0, 0.3), true, yaw);
    fixture.terrain.base.request_reset();
    let _ = wait_normal(&mut fixture, "normal retained front", OPAQUE, 0.3, rotated);
    fixture.set_terrain_depth(1.0);
    fixture.set_opaque(Vec3::new(0.0, 0.0, -1.0), true, yaw);
    fixture.terrain.base.request_reset();
    let _ = wait_normal(&mut fixture, "normal terrain front", TERRAIN, 1.0, Vec3::Z);

    // Removal removes retained normal coverage with retained depth/color.
    fixture.set_terrain_depth(0.0);
    fixture.hide_terrain(true);
    fixture.set_opaque(Vec3::new(0.0, 0.0, 0.05), true, 0.0);
    fixture.terrain.base.request_reset();
    let pre_removal = wait_normal(&mut fixture, "normal pre-removal", OPAQUE, 0.05, Vec3::Z);
    let background_normal = pixel(
        pre_removal.normal.as_ref().expect("pre-removal normal"),
        0,
        0,
    );
    assert!(normal_is(
        pixel(pre_removal.normal.as_ref().unwrap(), SIZE.x / 2, SIZE.y / 2),
        Vec3::Z
    ));
    fixture.release_retained();
    fixture.clear_retained();
    fixture.terrain.base.request_reset();
    let removed = fixture.wait("normal removal", |terrain| {
        let snap = snapshot(terrain)?;
        let normal = snap.normal.as_ref()?;
        let depth_zero = snap
            .depth
            .as_ref()
            .is_some_and(|d| d.iter().all(|p| p[0] == 0.0));
        let no_color = snap.source.iter().all(|p| !visible(*p));
        let background = normal
            .iter()
            .all(|p| (0..4).all(|c| approx(p[c], background_normal[c], 3.0e-3)));
        (depth_zero && no_color && background)
            .then_some(snap)
            .or_else(|| fail(terrain, "normal coverage removal and background restore"))
    });
    assert!(removed.normal.is_some());
    assert!(removed.source.iter().all(|p| !visible(*p)));

    // NormalPrepass off/on transition.
    fixture.set_opaque(Vec3::new(0.0, 0.0, 0.05), true, 0.0);
    fixture
        .terrain
        .base
        .app
        .world_mut()
        .entity_mut(camera)
        .remove::<NormalPrepass>();
    fixture.terrain.base.request_reset();
    let off = fixture.wait("normal off", |terrain| {
        let snap = snapshot(terrain)?;
        let center = pixel(&snap.source, SIZE.x / 2, SIZE.y / 2);
        let depth = pixel(snap.depth.as_ref()?, SIZE.x / 2, SIZE.y / 2)[0];
        (snap.normal.is_none()
            && color_is(center, OPAQUE, 0.025)
            && approx(depth, depth_far(0.05), 1.5e-3)
            && reset_executed(terrain, &snap))
        .then_some(snap)
        .or_else(|| fail(terrain, "normal off configuration"))
    });
    assert!(off.normal.is_none());
    fixture
        .terrain
        .base
        .app
        .world_mut()
        .entity_mut(camera)
        .insert(NormalPrepass);
    fixture.terrain.base.request_reset();
    let _ = wait_normal(&mut fixture, "normal restored", OPAQUE, 0.05, Vec3::Z);
    println!("PASS normal readback: encoding, rotation, cutout, overlap, removal, toggle");
}

fn pixel_index(x: u32, y: u32) -> usize {
    (y * SIZE.x + x) as usize
}
