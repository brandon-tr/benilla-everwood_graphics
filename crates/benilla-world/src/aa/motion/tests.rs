use super::*;
use bevy::render::camera::TemporalJitter;
use bevy::render::render_graph::{EmptyNode, RenderGraph, RenderLabel};
use bevy::render::sync_world::MainEntity;
use bevy::render::view::{ColorGrading, RetainedViewEntity};

#[cfg(test)]
fn finite(value: f32) -> bool {
    value.is_finite()
}

#[cfg(test)]
fn finite_vec4(value: Vec4) -> bool {
    value
        .to_array()
        .iter()
        .all(|component| component.is_finite())
}

#[cfg(test)]
fn finite_vec3(value: Vec3) -> bool {
    value
        .to_array()
        .iter()
        .all(|component| component.is_finite())
}

/// CPU oracle matching the shader: `uv` is current physical UV, `depth` current reverse depth.
/// Returns `(motion.xy, previous_depth)` or `None` for the same rejection branches.
#[cfg(test)]
fn reproject_rigid(uv: Vec2, depth: f32, params: &MotionUniform) -> Option<Vec3> {
    if !finite(uv.x) || !finite(uv.y) || !finite(depth) || depth <= 0.0 || depth > 1.0 {
        return None;
    }
    let ndc = Vec4::new(2.0 * uv.x - 1.0, 1.0 - 2.0 * uv.y, depth, 1.0);
    let current_h = params.current_view_from_clip * ndc;
    if !finite_vec4(current_h) || current_h.w.abs() < 1e-8 {
        return None;
    }
    let current_view = current_h.truncate() / current_h.w;
    let previous_h = params.previous_clip_from_current_view * current_view.extend(1.0);
    if !finite_vec4(previous_h) || previous_h.w <= 1e-8 {
        return None;
    }
    let previous_ndc = previous_h.truncate() / previous_h.w;
    if !finite_vec3(previous_ndc) || previous_ndc.z < 0.0 || previous_ndc.z > 1.0 {
        return None;
    }
    let previous_uv = Vec2::new(0.5 * previous_ndc.x + 0.5, 0.5 - 0.5 * previous_ndc.y);
    if !finite(previous_uv.x)
        || !finite(previous_uv.y)
        || previous_uv.x < 0.0
        || previous_uv.x > 1.0
        || previous_uv.y < 0.0
        || previous_uv.y > 1.0
    {
        return None;
    }
    Some(Vec3::new(
        uv.x - previous_uv.x - params.raster_jitter_delta.x,
        uv.y - previous_uv.y - params.raster_jitter_delta.y,
        previous_ndc.z,
    ))
}

use crate::view::world_fovy;

fn camera(eye: Vec3, target: Vec3, aspect: f32) -> CameraMatrices {
    let world_from_view = Mat4::look_at_rh(eye, target, Vec3::Y).inverse();
    let clip_from_view = Mat4::perspective_infinite_reverse_rh(world_fovy(aspect), aspect, 0.1);
    CameraMatrices {
        clip_from_view,
        world_from_view,
        jitter_uv: Vec2::ZERO,
    }
}

fn project(matrices: &CameraMatrices, point: Vec3) -> Option<(Vec2, f32)> {
    let view = matrices.world_from_view.inverse() * point.extend(1.0);
    let clip = matrices.clip_from_view * view;
    if clip.w <= 1e-8 || !finite_vec4(clip) {
        return None;
    }
    let ndc = clip.truncate() / clip.w;
    Some((Vec2::new(0.5 * ndc.x + 0.5, 0.5 - 0.5 * ndc.y), ndc.z))
}

fn motion_at(previous: CameraMatrices, current: CameraMatrices, point: Vec3) -> Option<Vec3> {
    let (uv, depth) = project(&current, point)?;
    let uniform = camera_reprojection(current, previous)?;
    reproject_rigid(uv, depth, &uniform)
}

#[derive(Clone, Copy, Debug)]
struct GateTap {
    reactive: f32,
    model: f32,
    depth: f32,
}

fn footprint(previous_uv: Vec2, extent: UVec2) -> [UVec2; 4] {
    let base = (previous_uv * extent.as_vec2() - Vec2::splat(0.5))
        .floor()
        .as_ivec2();
    let max = extent.as_ivec2() - IVec2::ONE;
    let offsets = [IVec2::ZERO, IVec2::X, IVec2::Y, IVec2::new(1, 1)];
    offsets.map(|offset| (base + offset).clamp(IVec2::ZERO, max).as_uvec2())
}

fn reconstruct_test(uv: Vec2, depth: f32, view_from_clip: Mat4) -> Option<Vec3> {
    let ndc = Vec4::new(2.0 * uv.x - 1.0, 1.0 - 2.0 * uv.y, depth, 1.0);
    let homogeneous = view_from_clip * ndc;
    if !finite_vec4(homogeneous) || homogeneous.w.abs() < 1e-8 {
        return None;
    }
    let view = homogeneous.truncate() / homogeneous.w;
    finite_vec3(view).then_some(view)
}

/// CPU mirror of the fragment's specified history gate. It is test evidence for the algorithm,
/// not evidence that the WGSL executed on a device.
#[allow(clippy::too_many_arguments)]
fn fused_gate(
    current_uv: Vec2,
    current_depth: f32,
    previous_uv: Vec2,
    expected_previous_depth: f32,
    params: &MotionUniform,
    extent: UVec2,
    current_reactive: f32,
    current_model: f32,
    taps: [GateTap; 4],
    dimensions_match: bool,
) -> Option<Vec4> {
    if !dimensions_match
        || !finite(current_uv.x)
        || !finite(current_uv.y)
        || !finite(current_depth)
        || current_depth <= 0.0
        || current_depth > 1.0
        || !finite(previous_uv.x)
        || !finite(previous_uv.y)
        || previous_uv.x < 0.0
        || previous_uv.x >= 1.0
        || previous_uv.y < 0.0
        || previous_uv.y >= 1.0
    {
        return None;
    }
    let ndc = Vec4::new(
        2.0 * current_uv.x - 1.0,
        1.0 - 2.0 * current_uv.y,
        current_depth,
        1.0,
    );
    let current_h = params.current_view_from_clip * ndc;
    if !finite_vec4(current_h) || current_h.w.abs() < 1e-8 {
        return None;
    }
    let current_view = current_h.truncate() / current_h.w;
    let previous_h = params.previous_clip_from_current_view * current_view.extend(1.0);
    if !finite_vec4(previous_h) || previous_h.w <= 1e-8 {
        return None;
    }
    let previous_ndc = previous_h.truncate() / previous_h.w;
    if !finite_vec3(previous_ndc) || previous_ndc.z < 0.0 || previous_ndc.z > 1.0 {
        return None;
    }
    let motion = current_uv - previous_uv - params.raster_jitter_delta.xy();
    if !finite(motion.x) || !finite(motion.y) {
        return None;
    }
    if !finite(current_reactive)
        || current_reactive != 0.0
        || !finite(current_model)
        || current_model != 0.0
    {
        return Some(Vec4::new(motion.x, motion.y, previous_ndc.z, 0.0));
    }
    let expected_view = reconstruct_test(
        previous_uv,
        expected_previous_depth,
        params.previous_view_from_clip,
    );
    let Some(expected_view) = expected_view else {
        return Some(Vec4::new(motion.x, motion.y, previous_ndc.z, 0.0));
    };
    let expected_distance = -expected_view.z;
    if !finite(expected_distance) || expected_distance <= 0.0 {
        return Some(Vec4::new(motion.x, motion.y, previous_ndc.z, 0.0));
    }
    for tap in footprint(previous_uv, extent).iter().zip(taps.iter()) {
        let (pixel, sample) = tap;
        if !finite(sample.reactive)
            || sample.reactive != 0.0
            || !finite(sample.model)
            || sample.model != 0.0
        {
            return Some(Vec4::new(motion.x, motion.y, previous_ndc.z, 0.0));
        }
        if !finite(sample.depth) || sample.depth <= 0.0 || sample.depth > 1.0 {
            return Some(Vec4::new(motion.x, motion.y, previous_ndc.z, 0.0));
        }
        let tap_uv = ((*pixel).as_vec2() + Vec2::splat(0.5)) / extent.as_vec2();
        let Some(actual_view) =
            reconstruct_test(tap_uv, sample.depth, params.previous_view_from_clip)
        else {
            return Some(Vec4::new(motion.x, motion.y, previous_ndc.z, 0.0));
        };
        let actual_distance = -actual_view.z;
        if !finite(actual_distance) || actual_distance <= 0.0 {
            return Some(Vec4::new(motion.x, motion.y, previous_ndc.z, 0.0));
        }
        let tolerance = f32::max(0.02, 0.005 * expected_distance);
        if (actual_distance - expected_distance).abs() > tolerance {
            return Some(Vec4::new(motion.x, motion.y, previous_ndc.z, 0.0));
        }
    }
    Some(Vec4::new(motion.x, motion.y, previous_ndc.z, 1.0))
}

fn gate_scene() -> (MotionUniform, UVec2, Vec2, f32, Vec2, f32, f32) {
    let frame = camera(Vec3::ZERO, Vec3::new(0.0, 0.0, -10.0), 16.0 / 9.0);
    let uniform = camera_reprojection(frame, frame).expect("valid uniform");
    let point = Vec3::new(0.0, 0.0, -10.0);
    let (uv, depth) = project(&frame, point).expect("visible point");
    let previous_view = frame.world_from_view.inverse() * point.extend(1.0);
    (
        uniform,
        UVec2::new(320, 200),
        uv,
        depth,
        uv,
        depth,
        -previous_view.z,
    )
}

fn clear_taps(depth: f32) -> [GateTap; 4] {
    [GateTap {
        reactive: 0.0,
        model: 0.0,
        depth,
    }; 4]
}

#[test]
fn uniform_layout_includes_previous_inverse_projection() {
    assert_eq!(MotionUniform::min_size().get(), 224);
    let current = camera(
        Vec3::new(1.0, 2.0, 3.0),
        Vec3::new(1.0, 2.0, -7.0),
        16.0 / 9.0,
    );
    let previous = camera(
        Vec3::new(0.5, 2.0, 3.5),
        Vec3::new(0.5, 2.0, -7.0),
        16.0 / 9.0,
    );
    let uniform = camera_reprojection(current, previous).expect("valid");
    assert_eq!(
        uniform.current_view_from_clip,
        current.clip_from_view.inverse()
    );
    assert_eq!(
        uniform.previous_view_from_clip,
        previous.clip_from_view.inverse()
    );
    assert!(finite_mat(uniform.previous_clip_from_current_view));
    assert_eq!(uniform.raster_jitter_delta, Vec4::ZERO);
}

#[test]
fn motion_encode_mode_fails_closed() {
    for inputs in [false, true] {
        for pipeline in [false, true] {
            for uniform in [false, true] {
                let expected = if inputs && pipeline && uniform {
                    MotionEncodeMode::Gated
                } else {
                    MotionEncodeMode::ClearOnly
                };
                assert_eq!(
                    motion_encode_mode(inputs, pipeline, uniform),
                    expected,
                    "inputs={inputs} pipeline={pipeline} uniform={uniform}"
                );
            }
        }
    }
}

#[test]
fn invalid_candidate_encoding_seeds_baseline() {
    let frame = camera(Vec3::ZERO, Vec3::new(0.0, 0.0, -10.0), 16.0 / 9.0);
    let mut frames = MotionFrames::new(frame);
    let publication = DepthPublication::new(7);
    assert_eq!(
        motion_encode_mode(false, false, false),
        MotionEncodeMode::ClearOnly
    );
    assert!(publication.encode(1, 7, || true));
    frames.advance(1, true, publication.is_current(1, 7));
    assert!(frames.previous_for_epoch(2).is_some());

    let mut missing = MotionFrames::new(frame);
    missing.advance(1, true, false);
    assert!(missing.previous_for_epoch(2).is_none());
}

#[test]
fn stationary_surface_passes_depth_and_clear_masks() {
    let (params, extent, uv, depth, previous_uv, previous_depth, expected_distance) = gate_scene();
    let result = fused_gate(
        uv,
        depth,
        previous_uv,
        previous_depth,
        &params,
        extent,
        0.0,
        0.0,
        clear_taps(previous_depth),
        true,
    )
    .expect("geometry valid");
    assert_eq!(result.w, 1.0);
    assert!((result.z - previous_depth).abs() < 1e-6);
    assert!(expected_distance > 0.0);
}

#[test]
fn closer_or_farther_previous_surface_is_rejected() {
    let (params, extent, uv, depth, previous_uv, expected_depth, expected_distance) = gate_scene();
    let frame = camera(Vec3::ZERO, Vec3::new(0.0, 0.0, -10.0), 16.0 / 9.0);
    let closer = project(&frame, Vec3::new(0.0, 0.0, -5.0)).unwrap().1;
    let farther = project(&frame, Vec3::new(0.0, 0.0, -20.0)).unwrap().1;
    assert!(expected_distance > 0.0);
    for previous_depth in [closer, farther, 0.0, f32::NAN] {
        let result = fused_gate(
            uv,
            depth,
            previous_uv,
            expected_depth,
            &params,
            extent,
            0.0,
            0.0,
            clear_taps(previous_depth),
            true,
        )
        .expect("geometry valid");
        assert_eq!(result.w, 0.0, "stored depth {previous_depth}");
    }
}

#[test]
fn masks_reject_current_and_previous_footprints() {
    let (params, extent, uv, depth, previous_uv, previous_depth, _) = gate_scene();
    for reactive in [0.0, 1.0] {
        for model in [0.0, 1.0] {
            if reactive == 0.0 && model == 0.0 {
                continue;
            }
            let result = fused_gate(
                uv,
                depth,
                previous_uv,
                previous_depth,
                &params,
                extent,
                reactive,
                model,
                clear_taps(previous_depth),
                true,
            )
            .unwrap();
            assert_eq!(result.w, 0.0, "current masks {reactive}/{model}");
        }
    }
    for tap_index in 0..4 {
        for lane in 0..2 {
            let mut taps = clear_taps(previous_depth);
            if lane == 0 {
                taps[tap_index].reactive = 1.0;
            } else {
                taps[tap_index].model = 1.0;
            }
            let result = fused_gate(
                uv,
                depth,
                previous_uv,
                previous_depth,
                &params,
                extent,
                0.0,
                0.0,
                taps,
                true,
            )
            .unwrap();
            assert_eq!(result.w, 0.0, "tap {tap_index} lane {lane}");
        }
    }
}

#[test]
fn reprojection_boundaries_reject_invalid_uv() {
    let (params, extent, uv, depth, _previous_uv, previous_depth, _) = gate_scene();
    for bad in [
        Vec2::new(-0.01, 0.5),
        Vec2::new(1.0, 0.5),
        Vec2::new(0.5, 1.0),
        Vec2::new(f32::NAN, 0.5),
    ] {
        assert!(
            fused_gate(
                uv,
                depth,
                bad,
                previous_depth,
                &params,
                extent,
                0.0,
                0.0,
                clear_taps(previous_depth),
                true,
            )
            .is_none(),
            "boundary {bad:?}"
        );
    }
    let clamped = footprint(Vec2::new(0.001, 0.999), extent);
    assert!(clamped
        .iter()
        .all(|pixel| pixel.x < extent.x && pixel.y < extent.y));
}

fn has_node_edge(graph: &RenderGraph, from: impl RenderLabel, to: impl RenderLabel) -> bool {
    let from = from.intern();
    let to = to.intern();
    graph
        .get_node_state(to)
        .map(|state| {
            state
                .edges
                .input_edges()
                .iter()
                .any(|edge| edge.get_output_node() == from && edge.get_input_node() == to)
        })
        .unwrap_or(false)
}

#[test]
fn motion_registration_follows_history_producers_and_precedes_end_main_pass() {
    use bevy::app::SubApp;
    use bevy::render::ExtractSchedule;

    let mut app = App::new();
    app.insert_resource(crate::view::MsaaSetting { samples: 1 })
        .init_resource::<crate::view::Viewer>()
        .init_resource::<crate::schedule::WorldLive>()
        .add_message::<crate::world_map::MapChange>();

    let mut render = SubApp::new();
    render.init_schedule(Render);
    render.init_schedule(RenderStartup);
    render.init_schedule(ExtractSchedule);
    let mut graph = RenderGraph::default();
    let mut core = RenderGraph::default();
    core.add_node(Node3d::MainOpaquePass, EmptyNode);
    core.add_node(Node3d::MainTransmissivePass, EmptyNode);
    core.add_node(Node3d::MainTransparentPass, EmptyNode);
    core.add_node(Node3d::EndMainPass, EmptyNode);
    graph.add_sub_graph(Core3d, core);
    render.insert_resource(graph);
    app.insert_sub_app(RenderApp, render);

    app.add_plugins(super::super::AaPlugin);

    let render_app = app.get_sub_app(RenderApp).expect("render app");
    let graph = render_app.world().resource::<RenderGraph>();
    let core = graph.get_sub_graph(Core3d).expect("Core3d subgraph");
    assert!(core.get_node_state(depth::TemporalDepthLabel).is_ok());
    assert!(core
        .get_node_state(super::super::reactive::TemporalReactiveLabel)
        .is_ok());
    assert!(core.get_node_state(TemporalMotionLabel).is_ok());
    assert!(has_node_edge(
        core,
        depth::TemporalDepthLabel,
        TemporalMotionLabel
    ));
    assert!(has_node_edge(
        core,
        super::super::reactive::TemporalReactiveLabel,
        TemporalMotionLabel
    ));
    assert!(has_node_edge(
        core,
        TemporalMotionLabel,
        Node3d::EndMainPass
    ));
    assert!(
        !has_node_edge(core, TemporalMotionLabel, Node3d::MainTransparentPass),
        "the obsolete before-transparent edge must be removed"
    );
}

#[test]
fn stationary_camera_has_zero_motion() {
    let frame = camera(Vec3::ZERO, Vec3::new(0.0, 0.0, -10.0), 16.0 / 9.0);
    let uniform = camera_reprojection(frame, frame).expect("valid");
    let (uv, depth) = project(&frame, Vec3::new(0.0, 0.0, -10.0)).expect("visible");
    let motion = reproject_rigid(uv, depth, &uniform).expect("reprojects");
    assert!(motion.x.abs() < 1e-6, "stationary x motion {}", motion.x);
    assert!(motion.y.abs() < 1e-6, "stationary y motion {}", motion.y);
    assert!((motion.z - depth).abs() < 1e-6);
    let near = project(&frame, Vec3::new(0.0, 0.0, -1.0)).expect("near").1;
    let far = project(&frame, Vec3::new(0.0, 0.0, -100.0)).expect("far").1;
    assert!(
        near > far,
        "reverse depth: near {near} must exceed far {far}"
    );
}

#[test]
fn camera_translation_has_current_minus_previous_uv_sign() {
    let previous = camera(Vec3::ZERO, Vec3::new(0.0, 0.0, -10.0), 16.0 / 9.0);
    let current = camera(Vec3::X, Vec3::new(1.0, 0.0, -10.0), 16.0 / 9.0);
    let point = Vec3::new(0.0, 0.0, -10.0);
    let (current_uv, current_depth) = project(&current, point).expect("visible now");
    let (previous_uv, _) = project(&previous, point).expect("visible before");
    let uniform = camera_reprojection(current, previous).expect("valid");
    let motion = reproject_rigid(current_uv, current_depth, &uniform).expect("reprojects");
    let expected = current_uv - previous_uv;
    assert!((motion.x - expected.x).abs() < 1e-5);
    assert!((motion.y - expected.y).abs() < 1e-5);
    assert!(
        motion.x < 0.0,
        "moving +X makes a stationary point drift left"
    );
}

#[test]
fn camera_rotation_matches_direct_projection() {
    let previous = camera(Vec3::ZERO, Vec3::new(0.0, 0.0, -10.0), 16.0 / 9.0);
    let current = camera(Vec3::ZERO, Vec3::new(-1.0, 0.0, -10.0), 16.0 / 9.0);
    let point = Vec3::new(0.0, 0.0, -10.0);
    let (current_uv, current_depth) = project(&current, point).expect("visible now");
    let (previous_uv, _) = project(&previous, point).expect("visible before");
    let uniform = camera_reprojection(current, previous).expect("valid");
    let motion = reproject_rigid(current_uv, current_depth, &uniform).expect("reprojects");
    let expected = current_uv - previous_uv;
    assert!((motion.x - expected.x).abs() < 1e-4);
    assert!((motion.y - expected.y).abs() < 1e-4);
}

#[test]
fn large_world_translation_preserves_relative_reprojection() {
    let point = Vec3::new(0.0, 0.0, -10.0);
    let near_previous = camera(Vec3::ZERO, Vec3::new(0.0, 0.0, -10.0), 16.0 / 9.0);
    let near_current = camera(Vec3::X, Vec3::new(1.0, 0.0, -10.0), 16.0 / 9.0);
    let offset = Vec3::new(9000.0, 0.0, 0.0);
    let far_previous = camera(offset, offset + Vec3::new(0.0, 0.0, -10.0), 16.0 / 9.0);
    let far_current = camera(
        offset + Vec3::X,
        offset + Vec3::new(1.0, 0.0, -10.0),
        16.0 / 9.0,
    );
    let near = motion_at(near_previous, near_current, point).expect("near");
    let far = motion_at(far_previous, far_current, point + offset).expect("far");
    assert!((near.x - far.x).abs() < 2e-3, "near {near:?} far {far:?}");
    assert!((near.y - far.y).abs() < 2e-3);
}

#[test]
fn motion_descriptor_is_full_size_single_sample_rgba16float() {
    let descriptor = motion_descriptor(UVec2::new(800, 600));
    assert_eq!(descriptor.size.width, 800);
    assert_eq!(descriptor.size.height, 600);
    assert_eq!(descriptor.size.depth_or_array_layers, 1);
    assert_eq!(descriptor.mip_level_count, 1);
    assert_eq!(descriptor.sample_count, 1);
    assert_eq!(descriptor.dimension, TextureDimension::D2);
    assert_eq!(descriptor.format, TextureFormat::Rgba16Float);
    assert!(descriptor.usage.contains(TextureUsages::RENDER_ATTACHMENT));
    assert!(descriptor.usage.contains(TextureUsages::TEXTURE_BINDING));
}

#[test]
fn background_offscreen_behind_camera_and_nonfinite_reprojection_are_invalid() {
    let frame = camera(Vec3::ZERO, Vec3::new(0.0, 0.0, -10.0), 16.0 / 9.0);
    let uniform = camera_reprojection(frame, frame).expect("valid");
    let uv = Vec2::splat(0.5);
    assert!(reproject_rigid(uv, 0.0, &uniform).is_none(), "background");
    assert!(
        reproject_rigid(Vec2::new(f32::NAN, 0.5), 0.5, &uniform).is_none(),
        "non-finite uv"
    );
    assert!(
        reproject_rigid(uv, f32::NAN, &uniform).is_none(),
        "non-finite depth"
    );

    // The point is centred now but was outside the previous camera's frame.
    let previous = camera(Vec3::ZERO, Vec3::new(0.0, 0.0, -10.0), 16.0 / 9.0);
    let current = camera(
        Vec3::new(100.0, 0.0, 0.0),
        Vec3::new(100.0, 0.0, -10.0),
        16.0 / 9.0,
    );
    let point = Vec3::new(100.0, 0.0, -10.0);
    let (current_uv, current_depth) = project(&current, point).expect("visible now");
    let uniform = camera_reprojection(current, previous).expect("valid");
    assert!(
        reproject_rigid(current_uv, current_depth, &uniform).is_none(),
        "previous UV is off-screen"
    );

    // The current camera looks +Z at the point, which is behind the previous camera.
    let previous = camera(Vec3::ZERO, Vec3::new(0.0, 0.0, -10.0), 16.0 / 9.0);
    let current = camera(Vec3::ZERO, Vec3::new(0.0, 0.0, 10.0), 16.0 / 9.0);
    let point = Vec3::new(0.0, 0.0, 10.0);
    let (current_uv, current_depth) = project(&current, point).expect("visible now");
    let uniform = camera_reprojection(current, previous).expect("valid");
    assert!(
        reproject_rigid(current_uv, current_depth, &uniform).is_none(),
        "previous clip is behind the camera"
    );
}

#[test]
fn first_encoded_frame_seeds_baseline_without_valid_motion() {
    let frame = camera(Vec3::ZERO, Vec3::new(0.0, 0.0, -10.0), 16.0 / 9.0);
    let frames = MotionFrames::new(frame);
    let (uniform, previous) = frame_motion(&frames, 1, UVec2::new(800, 600));
    assert!(previous.is_none());
    assert_eq!(uniform.extent_and_flags.z, 0.0);
    let mut frames = frames;
    frames.advance(1, true, true);
    assert!(frames.previous_for_epoch(2).is_some());
}

#[test]
fn only_consecutive_encoded_frames_supply_previous_matrices() {
    let frame = camera(Vec3::ZERO, Vec3::new(0.0, 0.0, -10.0), 16.0 / 9.0);
    let mut frames = MotionFrames::new(frame);
    frames.advance(1, true, true);
    assert!(frames.previous_for_epoch(2).is_some());
    assert!(
        frames.previous_for_epoch(3).is_none(),
        "a gap is not consecutive"
    );
}

#[test]
fn skipped_depth_or_motion_resets_baseline() {
    let frame = camera(Vec3::ZERO, Vec3::new(0.0, 0.0, -10.0), 16.0 / 9.0);
    let mut frames = MotionFrames::new(frame);
    frames.advance(1, true, true);
    frames.advance(2, false, true);
    assert!(frames.previous_for_epoch(3).is_none());
    frames.advance(2, true, true);
    frames.advance(3, true, false);
    assert!(frames.previous_for_epoch(4).is_none());
}

#[test]
fn generation_resize_disable_and_view_identity_do_not_share_baselines() {
    let a = DepthKey {
        main_view: Entity::from_raw_u32(1).unwrap(),
        generation: 1,
        size: UVec2::new(800, 600),
    };
    let b = DepthKey {
        main_view: Entity::from_raw_u32(2).unwrap(),
        ..a
    };
    let c = DepthKey { generation: 2, ..a };
    let d = DepthKey {
        size: UVec2::new(640, 480),
        ..a
    };
    assert_ne!(a, b);
    assert_ne!(a, c);
    assert_ne!(a, d);
    let frame = camera(Vec3::ZERO, Vec3::new(0.0, 0.0, -10.0), 16.0 / 9.0);
    let frames = MotionFrames::new(frame);
    assert!(frames.previous.is_none());
}

#[test]
fn motion_producer_does_not_require_prior_motion_publication() {
    let publication = DepthPublication::new(7);
    assert!(!publication.is_current(1, 7));
    assert!(publication.encode(1, 7, || true));
    assert!(publication.is_current(1, 7));
}

#[test]
fn motion_consumer_rejects_stale_depth_publication() {
    let publication = DepthPublication::new(7);
    assert!(publication.encode(1, 7, || true));
    assert!(publication.is_current(1, 7));
    assert!(!publication.is_current(2, 7));
    assert!(!publication.is_current(1, 8));
}

#[test]
fn headless_aa_plugin_still_runs() {
    let mut app = App::new();
    app.insert_resource(crate::view::MsaaSetting { samples: 1 })
        .init_resource::<crate::view::Viewer>()
        .init_resource::<crate::schedule::WorldLive>()
        .add_message::<crate::world_map::MapChange>()
        .add_plugins(super::super::AaPlugin);
    app.update();
}

fn extracted_test_view(
    eye: Vec3,
    target: Vec3,
    aspect: f32,
    size: UVec2,
    offset: Option<Vec2>,
) -> (ExtractedView, Option<TemporalJitter>) {
    let world_from_view = Mat4::look_at_rh(eye, target, Vec3::Y).inverse();
    let view = ExtractedView {
        retained_view_entity: RetainedViewEntity::new(
            MainEntity::from(Entity::PLACEHOLDER),
            None,
            0,
        ),
        clip_from_view: Mat4::perspective_infinite_reverse_rh(world_fovy(aspect), aspect, 0.1),
        world_from_view: GlobalTransform::from(world_from_view),
        clip_from_world: None,
        hdr: true,
        viewport: UVec4::new(0, 0, size.x, size.y),
        color_grading: ColorGrading::default(),
        invert_culling: false,
    };
    (view, offset.map(|offset| TemporalJitter { offset }))
}

fn jittered_camera(
    eye: Vec3,
    target: Vec3,
    aspect: f32,
    size: UVec2,
    offset: Vec2,
) -> CameraMatrices {
    let (view, jitter) = extracted_test_view(eye, target, aspect, size, Some(offset));
    CameraMatrices::from_view(&view, jitter.as_ref(), size).expect("supported jitter")
}

#[test]
fn jittered_snapshot_matches_bevy_projection() {
    let size = UVec2::new(320, 200);
    let offset = Vec2::new(0.25, -0.4);
    let (view, jitter) = extracted_test_view(
        Vec3::ZERO,
        Vec3::new(0.0, 0.0, -10.0),
        16.0 / 9.0,
        size,
        Some(offset),
    );
    let unjittered = view.clip_from_view;
    let snapshot = CameraMatrices::from_view(&view, jitter.as_ref(), size).expect("supported");
    let mut expected = unjittered;
    jitter
        .as_ref()
        .unwrap()
        .jitter_projection(&mut expected, size.as_vec2());
    assert_eq!(snapshot.clip_from_view, expected);
    assert_eq!(
        view.clip_from_view, unjittered,
        "source projection must not mutate"
    );
    assert_eq!(snapshot.jitter_uv, offset / size.as_vec2());
}

#[test]
fn stationary_jitter_has_zero_unjittered_motion() {
    let size = UVec2::new(320, 200);
    let eye = Vec3::new(0.5, -0.25, 1.0);
    let target = Vec3::new(0.0, 0.0, -10.0);
    for (current_offset, previous_offset) in [
        (Vec2::new(0.4, -0.25), Vec2::new(-0.3, 0.35)),
        (Vec2::new(-0.5, 0.5), Vec2::new(0.5, -0.5)),
    ] {
        let current = jittered_camera(eye, target, 16.0 / 9.0, size, current_offset);
        let previous = jittered_camera(eye, target, 16.0 / 9.0, size, previous_offset);
        let uniform = camera_reprojection(current, previous).expect("valid reprojection");
        let expected_delta = previous_offset / size.as_vec2() - current_offset / size.as_vec2();
        assert!(
            (uniform.raster_jitter_delta.xy() - expected_delta).length() < 1.0e-6,
            "uniform jitter delta {:?} vs {expected_delta:?}",
            uniform.raster_jitter_delta.xy()
        );
        for point in [
            Vec3::new(0.0, 0.0, -10.0),
            Vec3::new(2.0, 1.0, -15.0),
            Vec3::new(-1.5, 0.5, -6.0),
        ] {
            let (current_uv, depth) = project(&current, point).expect("current visible");
            let (previous_uv, _) = project(&previous, point).expect("previous visible");
            let raw = current_uv - previous_uv;
            assert!(
                raw.length() > 1.0e-4,
                "raster coordinates must move with jitter: {raw:?}"
            );
            let corrected = reproject_rigid(current_uv, depth, &uniform).expect("reprojects");
            assert!(
                corrected.truncate().length() < 2.0e-5,
                "stationary corrected motion {:?} from raw {raw:?}",
                corrected.truncate()
            );
        }
    }
}

#[test]
fn moving_camera_with_jitter_matches_unjittered_motion() {
    let size = UVec2::new(320, 200);
    let aspect = 16.0 / 9.0;
    let current_eye = Vec3::new(0.5, 0.2, 0.5);
    let previous_eye = Vec3::new(-0.25, 0.0, 0.0);
    let target = Vec3::new(0.0, 0.0, -10.0);
    let jittered_current =
        jittered_camera(current_eye, target, aspect, size, Vec2::new(0.35, -0.3));
    let jittered_previous =
        jittered_camera(previous_eye, target, aspect, size, Vec2::new(-0.4, 0.2));
    let unjittered_current = camera(current_eye, target, aspect);
    let unjittered_previous = camera(previous_eye, target, aspect);
    let uniform = camera_reprojection(jittered_current, jittered_previous).expect("valid");
    for point in [
        Vec3::new(0.0, 0.0, -10.0),
        Vec3::new(2.0, 1.0, -15.0),
        Vec3::new(-1.0, -0.5, -7.0),
    ] {
        let (current_uv, depth) = project(&jittered_current, point).expect("current");
        let (expected_current_uv, _) =
            project(&unjittered_current, point).expect("unjittered current");
        let (expected_previous_uv, _) =
            project(&unjittered_previous, point).expect("unjittered previous");
        let expected = expected_current_uv - expected_previous_uv;
        let corrected = reproject_rigid(current_uv, depth, &uniform).expect("reprojects");
        assert!(
            (corrected.truncate() - expected).length() < 2.0e-5,
            "jittered motion {:?} must match independent {expected:?}",
            corrected.truncate()
        );
    }
}

#[test]
fn jittered_previous_depth_uses_raster_coordinates() {
    let size = UVec2::new(64, 64);
    let eye = Vec3::ZERO;
    let target = Vec3::new(0.0, 0.0, -10.0);
    let current = jittered_camera(eye, target, 1.0, size, Vec2::new(0.5, 0.0));
    let previous = jittered_camera(eye, target, 1.0, size, Vec2::new(-0.5, 0.0));
    let point = Vec3::new(0.0, 0.0, -10.0);
    let (current_uv, current_depth) = project(&current, point).expect("current");
    let (raw_previous_uv, previous_depth) = project(&previous, point).expect("previous");
    let (unjittered_uv, _) = project(&camera(eye, target, 1.0), point).expect("unjittered");
    assert_ne!(raw_previous_uv, unjittered_uv);
    let raw_footprint = footprint(raw_previous_uv, size);
    let unjittered_footprint = footprint(unjittered_uv, size);
    assert_ne!(raw_footprint, unjittered_footprint);

    let poison = unjittered_footprint
        .iter()
        .copied()
        .find(|pixel| !raw_footprint.contains(pixel))
        .expect("the wrong footprint must contain a pixel absent from the raw footprint");
    assert!(
        !raw_footprint.contains(&poison),
        "poisoned pixel must be exclusive to the wrong footprint"
    );
    let spatial = spatial_taps(size, previous_depth, poison);

    let uniform = camera_reprojection(current, previous).expect("valid");
    let corrected = reproject_rigid(current_uv, current_depth, &uniform).expect("reprojects");
    let derived_previous_raw = current_uv - corrected.truncate() - uniform.raster_jitter_delta.xy();
    assert!(
        (derived_previous_raw - raw_previous_uv).length() < 1.0e-6,
        "derived raw UV {derived_previous_raw:?} != projected {raw_previous_uv:?}"
    );

    let raw_gate = fused_gate(
        current_uv,
        current_depth,
        derived_previous_raw,
        previous_depth,
        &uniform,
        size,
        0.0,
        0.0,
        taps_at(&spatial, raw_footprint, size),
        true,
    );
    assert_eq!(raw_gate.expect("raw footprint passes").w, 1.0);

    let wrong_gate = fused_gate(
        current_uv,
        current_depth,
        unjittered_uv,
        previous_depth,
        &uniform,
        size,
        0.0,
        0.0,
        taps_at(&spatial, unjittered_footprint, size),
        true,
    );
    assert_eq!(
        wrong_gate.expect("wrong footprint is encoded").w,
        0.0,
        "the wrong footprint must see the shared poisoned model tap"
    );
}

fn spatial_taps(size: UVec2, depth: f32, poison: UVec2) -> Vec<GateTap> {
    let mut taps = vec![
        GateTap {
            reactive: 0.0,
            model: 0.0,
            depth,
        };
        (size.x * size.y) as usize
    ];
    taps[(poison.y * size.x + poison.x) as usize].model = 1.0;
    taps
}

fn taps_at(spatial: &[GateTap], footprint: [UVec2; 4], size: UVec2) -> [GateTap; 4] {
    footprint.map(|pixel| spatial[(pixel.y * size.x + pixel.x) as usize])
}

#[test]
fn unsupported_jitter_fails_closed() {
    let size = UVec2::new(320, 200);
    let eye = Vec3::ZERO;
    let target = Vec3::new(0.0, 0.0, -10.0);

    for offset in [
        Vec2::new(f32::NAN, 0.0),
        Vec2::new(0.0, f32::INFINITY),
        Vec2::new(0.5001, 0.0),
        Vec2::new(0.0, -0.5001),
    ] {
        let (view, jitter) = extracted_test_view(eye, target, 16.0 / 9.0, size, Some(offset));
        assert!(CameraMatrices::from_view(&view, jitter.as_ref(), size).is_none());
    }

    let (view, jitter) =
        extracted_test_view(eye, target, 16.0 / 9.0, size, Some(Vec2::new(0.1, 0.1)));
    assert!(CameraMatrices::from_view(&view, jitter.as_ref(), UVec2::ZERO).is_none());

    let (mut view, jitter) =
        extracted_test_view(eye, target, 16.0 / 9.0, size, Some(Vec2::new(0.1, 0.1)));
    view.viewport = UVec4::new(1, 0, size.x, size.y);
    assert!(CameraMatrices::from_view(&view, jitter.as_ref(), size).is_none());

    let (mut view, jitter) =
        extracted_test_view(eye, target, 16.0 / 9.0, size, Some(Vec2::new(0.1, 0.1)));
    view.clip_from_view = Mat4::orthographic_rh(-1.0, 1.0, -1.0, 1.0, 0.1, 100.0);
    assert!(CameraMatrices::from_view(&view, jitter.as_ref(), size).is_none());

    let (mut view, jitter) =
        extracted_test_view(eye, target, 16.0 / 9.0, size, Some(Vec2::new(0.1, 0.1)));
    view.clip_from_world = Some(Mat4::IDENTITY);
    assert!(CameraMatrices::from_view(&view, jitter.as_ref(), size).is_none());
}
