use super::*;
use bevy::render::render_graph::{EmptyNode, RenderGraph};

fn key(main: u32, generation: u64, size: UVec2) -> DepthKey {
    DepthKey {
        main_view: Entity::from_raw_u32(main).unwrap(),
        generation,
        size,
    }
}

#[test]
fn resolve_decision_tracks_marker_source_and_key() {
    let requested = key(1, 2, UVec2::new(64, 64));
    assert_eq!(
        resolve_decision(false, Some(requested), requested, true),
        ResolveDecision::Remove
    );
    assert_eq!(
        resolve_decision(true, Some(requested), requested, false),
        ResolveDecision::Remove
    );
    assert_eq!(
        resolve_decision(true, Some(requested), requested, true),
        ResolveDecision::Keep
    );
    for changed in [
        key(2, 2, requested.size),
        key(1, 3, requested.size),
        key(1, 2, UVec2::new(48, 32)),
    ] {
        assert_eq!(
            resolve_decision(true, Some(changed), requested, true),
            ResolveDecision::Replace
        );
    }
}

#[test]
fn resolve_mode_prefers_temporal_and_fails_closed() {
    assert_eq!(
        resolve_mode(false, true, true, true, true, true),
        ResolveMode::Skip
    );
    assert_eq!(
        resolve_mode(true, false, true, true, true, true),
        ResolveMode::Skip
    );
    assert_eq!(
        resolve_mode(true, true, true, false, true, true),
        ResolveMode::Seed
    );
    assert_eq!(
        resolve_mode(true, true, false, true, true, true),
        ResolveMode::Seed
    );
    assert_eq!(
        resolve_mode(true, true, true, true, true, true),
        ResolveMode::Temporal
    );
    assert_eq!(
        resolve_mode(true, true, true, true, false, true),
        ResolveMode::Temporal
    );
    assert_eq!(
        resolve_mode(true, true, true, true, true, false),
        ResolveMode::Skip
    );
    assert_eq!(
        resolve_mode(true, true, false, false, false, true),
        ResolveMode::Skip
    );
}

fn history_tap_raw(history_tap: Vec2, previous_raw: Vec2, history_uv: Vec2) -> Vec2 {
    history_tap + (previous_raw - history_uv)
}

fn candidate_pixels(raw_uv: Vec2, extent: UVec2) -> [UVec2; 4] {
    let base = (raw_uv * extent.as_vec2()).floor().as_uvec2();
    let max = extent - UVec2::ONE;
    [
        base.min(max),
        (base + UVec2::X).min(max),
        (base + UVec2::Y).min(max),
        (base + UVec2::ONE).min(max),
    ]
}

fn support_pixels(raw_uv: Vec2, extent: UVec2) -> [UVec2; 4] {
    // CPU model of support_ok's clamped bilinear footprint, not shader execution.
    let base = (raw_uv * extent.as_vec2() - Vec2::splat(0.5))
        .floor()
        .as_uvec2();
    let max = extent - UVec2::ONE;
    [
        base.min(max),
        (base + UVec2::X).min(max),
        (base + UVec2::Y).min(max),
        (base + UVec2::ONE).min(max),
    ]
}

fn support_rejects(extent: UVec2, raw_uv: Vec2, reactive: &[f32], model: &[f32]) -> bool {
    let index = |pixel: UVec2| (pixel.y * extent.x + pixel.x) as usize;
    support_pixels(raw_uv, extent)
        .iter()
        .any(|pixel| reactive[index(*pixel)] != 0.0 || model[index(*pixel)] != 0.0)
}

fn current_rejects(extent: UVec2, raw_uv: Vec2, reactive: &[f32], model: &[f32]) -> bool {
    let index = |pixel: UVec2| (pixel.y * extent.x + pixel.x) as usize;
    candidate_pixels(raw_uv, extent)
        .iter()
        .any(|pixel| reactive[index(*pixel)] != 0.0 || model[index(*pixel)] != 0.0)
}

#[test]
fn filter_support_uses_half_texel_and_shared_spatial_masks() {
    let extent = UVec2::splat(8);
    let cells = (extent.x * extent.y) as usize;
    let clear_reactive = vec![0.0; cells];
    let mut clear_model = vec![0.0; cells];

    let raw = Vec2::new(2.25, 2.25) / extent.as_vec2();
    let naive = (raw * extent.as_vec2()).floor().as_uvec2();
    let base = (raw * extent.as_vec2() - Vec2::splat(0.5))
        .floor()
        .as_uvec2();
    assert_ne!(naive, base, "fraction 0.25 must move left/top support");
    assert_eq!(base, UVec2::new(1, 1));
    assert_eq!(naive, UVec2::new(2, 2));

    assert!(!support_rejects(extent, raw, &clear_reactive, &clear_model));
    let candidate = candidate_pixels(raw, extent);
    let support = support_pixels(raw, extent);
    let extra = support
        .iter()
        .copied()
        .find(|pixel| !candidate.contains(pixel))
        .expect("support must extend beyond the candidate footprint");
    clear_model[(extra.y * extent.x + extra.x) as usize] = 1.0;
    assert!(
        support_rejects(extent, raw, &clear_reactive, &clear_model),
        "a rejected extra filter-support tap must reject"
    );
    clear_model[(extra.y * extent.x + extra.x) as usize] = 0.0;
    assert!(!support_rejects(extent, raw, &clear_reactive, &clear_model));

    let outside = UVec2::new(7, 7);
    assert!(!support.contains(&outside));
    clear_model[(outside.y * extent.x + outside.x) as usize] = 1.0;
    assert!(
        !support_rejects(extent, raw, &clear_reactive, &clear_model),
        "a poison outside actual filter support must not reject"
    );
    clear_model[(outside.y * extent.x + outside.x) as usize] = 0.0;

    let current = candidate[0];
    let mut current_reactive = clear_reactive.clone();
    current_reactive[(current.y * extent.x + current.x) as usize] = 1.0;
    assert!(current_rejects(
        extent,
        raw,
        &current_reactive,
        &clear_model
    ));
    let mut current_model = clear_model.clone();
    current_model[(current.y * extent.x + current.x) as usize] = 1.0;
    assert!(current_rejects(
        extent,
        raw,
        &clear_reactive,
        &current_model
    ));

    let mapped = history_tap_raw(
        Vec2::new(0.52, 0.48),
        Vec2::new(0.70, 0.30),
        Vec2::new(0.55, 0.45),
    );
    assert!((mapped - Vec2::new(0.67, 0.33)).length() < 1.0e-7);
    assert!(!support_rejects(
        extent,
        Vec2::ZERO,
        &clear_reactive,
        &clear_model
    ));
    assert!(!support_rejects(
        extent,
        Vec2::splat(0.999),
        &clear_reactive,
        &clear_model
    ));
}

fn next_confidence(minimum: f32, stationary: bool) -> f32 {
    if stationary {
        (minimum + 10.0).min(1.0 / 0.015)
    } else {
        1.0
    }
}

fn safe_history(sample: [f32; 4]) -> bool {
    sample.iter().all(|v| v.is_finite())
        && sample[3] > 0.0
        && (0..3).all(|channel| sample[channel] >= 0.0 && sample[channel] < 1.0)
}

fn guarded_inverse(color: [f32; 3]) -> Option<[f32; 3]> {
    let max = color[0].max(color[1]).max(color[2]);
    let denominator = 1.0 - max;
    if !denominator.is_finite() || denominator.abs() <= 1.0e-8 {
        return None;
    }
    let result = color.map(|channel| channel / denominator);
    result.iter().all(|v| v.is_finite()).then_some(result)
}

#[test]
fn confidence_and_numeric_guards_match_shader_contract() {
    assert_eq!(next_confidence(1.0, true), 11.0);
    assert_eq!(next_confidence(11.0, true), 21.0);
    assert_eq!(next_confidence(5.0, true), 15.0);
    assert_eq!(next_confidence(21.0, false), 1.0);
    assert!(safe_history([0.2, 0.3, 0.4, 11.0]));
    assert!(!safe_history([0.2, 0.3, 0.4, 0.0]));
    assert!(!safe_history([f32::NAN, 0.3, 0.4, 1.0]));
    assert!(!safe_history([1.0, 0.3, 0.4, 1.0]));
    assert!(guarded_inverse([0.2, 0.3, 0.4]).is_some());
    assert!(guarded_inverse([1.0, 0.3, 0.4]).is_none());
    assert!(guarded_inverse([f32::NAN, 0.3, 0.4]).is_none());
}

#[test]
fn temporal_layout_binds_r32float_depth_and_shader_agrees() {
    let entries = temporal_layout_entries();
    for binding in [3, 4] {
        match entries[binding as usize].ty {
            BindingType::Texture {
                sample_type: TextureSampleType::Float { filterable: false },
                view_dimension: TextureViewDimension::D2,
                multisampled: false,
            } => {}
            other => panic!("binding {binding} must be nonfiltering R32Float color: {other:?}"),
        }
    }
    let shader = include_str!("../../shaders/temporal_resolve.wgsl");
    assert!(shader.contains("@binding(3) var current_depth: texture_2d<f32>;"));
    assert!(shader.contains("@binding(4) var previous_depth: texture_2d<f32>;"));
    assert!(shader.contains("textureLoad(current_depth, pixel, 0).x"));
}

#[test]
fn resolve_registration_follows_scene_color_and_precedes_end() {
    use bevy::app::SubApp;
    use bevy::render::render_graph::RenderGraph;
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
    let core = graph.get_sub_graph(Core3d).expect("Core3d");
    assert!(core.get_node_state(TemporalResolveLabel).is_ok());
    assert!(has_edge(
        core,
        TemporalSceneColorLabel,
        TemporalResolveLabel
    ));
    assert!(has_edge(core, TemporalResolveLabel, Node3d::EndMainPass));
}

fn has_edge(graph: &RenderGraph, from: impl RenderLabel, to: impl RenderLabel) -> bool {
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
fn headless_resolve_plugin_runs() {
    let mut app = App::new();
    app.insert_resource(crate::view::MsaaSetting { samples: 1 })
        .init_resource::<crate::view::Viewer>()
        .init_resource::<crate::schedule::WorldLive>()
        .add_message::<crate::world_map::MapChange>()
        .add_plugins(super::super::AaPlugin);
    app.update();
}
