use super::*;
use crate::aa::reactive::TemporalReactiveLabel;
use bevy::render::render_graph::RenderGraph;

fn key(main_view: u32, generation: u64, size: UVec2) -> DepthKey {
    DepthKey {
        main_view: Entity::from_raw_u32(main_view).unwrap(),
        generation,
        size,
    }
}

#[test]
fn scene_color_descriptor_is_native_rgba16float() {
    let descriptor = scene_color_descriptor(UVec2::new(320, 200));
    assert_eq!(descriptor.size.width, 320);
    assert_eq!(descriptor.size.height, 200);
    assert_eq!(descriptor.size.depth_or_array_layers, 1);
    assert_eq!(descriptor.mip_level_count, 1);
    assert_eq!(descriptor.sample_count, 1);
    assert_eq!(descriptor.dimension, TextureDimension::D2);
    assert_eq!(descriptor.format, TextureFormat::Rgba16Float);
    assert!(descriptor
        .usage
        .contains(TextureUsages::RENDER_ATTACHMENT | TextureUsages::TEXTURE_BINDING));
}

#[test]
fn matching_scene_key_reuses_pair() {
    let requested = key(1, 2, UVec2::new(320, 200));
    assert_eq!(
        scene_color_decision(Some(requested), requested, true),
        SceneColorDecision::Keep
    );
}

#[test]
fn scene_key_changes_replace_both_slots() {
    let requested = key(1, 2, UVec2::new(320, 200));
    for existing in [
        DepthKey {
            main_view: Entity::from_raw_u32(2).unwrap(),
            ..requested
        },
        DepthKey {
            generation: 3,
            ..requested
        },
        DepthKey {
            size: UVec2::new(640, 400),
            ..requested
        },
    ] {
        assert_eq!(
            scene_color_decision(Some(existing), requested, true),
            SceneColorDecision::Replace
        );
    }
}

#[test]
fn missing_scene_source_retires_pair() {
    let requested = key(1, 2, UVec2::new(320, 200));
    assert_eq!(
        scene_color_decision(Some(requested), requested, false),
        SceneColorDecision::Remove
    );
}

#[test]
fn scene_capture_can_encode_without_previous() {
    let frames = TemporalFramePair::new(4, [10, 20]);
    assert!(frames.encode_current(1, 4, |slot| {
        assert_eq!(*slot, 20);
        true
    }));
    assert_eq!(frames.current(1, 4), Some(&20));
    assert!(frames.previous(1, 4).is_none());
}

#[test]
fn consecutive_scene_captures_retain_previous() {
    let frames = TemporalFramePair::new(4, [10, 20]);
    let destination = std::cell::RefCell::new(Vec::new());
    assert!(frames.encode_current(1, 4, |slot| {
        destination.borrow_mut().push(*slot);
        true
    }));
    assert!(frames.encode_current(2, 4, |slot| {
        destination.borrow_mut().push(*slot);
        true
    }));
    assert_eq!(frames.current(2, 4), Some(&10));
    assert_eq!(frames.previous(2, 4), Some(&20));
    assert!(frames.encode_current(3, 4, |slot| {
        destination.borrow_mut().push(*slot);
        true
    }));
    assert_eq!(*destination.borrow(), vec![20, 10, 20]);
    assert_eq!(frames.current(3, 4), Some(&20));
    assert_eq!(frames.previous(3, 4), Some(&10));
}

#[test]
fn skipped_scene_capture_rejects_previous() {
    let frames = TemporalFramePair::new(4, [10, 20]);
    assert!(frames.encode_current(1, 4, |_| true));
    assert!(!frames.encode_current(2, 4, |_| false));
    assert!(frames.encode_current(3, 4, |_| true));
    assert!(frames.current(3, 4).is_some());
    assert!(frames.previous(3, 4).is_none());
    assert!(frames.encode_current(4, 4, |_| true));
    assert!(frames.previous(4, 4).is_some());
}

#[test]
fn replacement_scene_capture_starts_empty() {
    let old = TemporalFramePair::new(3, [10, 20]);
    assert!(old.encode_current(1, 3, |_| true));
    let fresh = TemporalFramePair::new(4, [10, 20]);
    assert!(fresh.current(1, 4).is_none());
    assert!(fresh.previous(1, 4).is_none());
    assert!(fresh.current(1, 3).is_none());
}

#[test]
fn independent_scene_views_do_not_share_publication() {
    let a = TemporalFramePair::new(4, [10, 20]);
    let b = TemporalFramePair::new(4, [10, 20]);
    assert!(a.encode_current(1, 4, |_| true));
    assert!(a.current(1, 4).is_some());
    assert!(b.current(1, 4).is_none());
    assert!(b.previous(1, 4).is_none());
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
fn scene_capture_registration_follows_motion_and_precedes_end() {
    use bevy::app::SubApp;
    use bevy::render::render_graph::{EmptyNode, RenderGraph};
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
    assert!(core.get_node_state(TemporalMotionLabel).is_ok());
    assert!(core.get_node_state(TemporalReactiveLabel).is_ok());
    assert!(core.get_node_state(TemporalSceneColorLabel).is_ok());
    assert!(has_node_edge(
        core,
        TemporalMotionLabel,
        TemporalSceneColorLabel
    ));
    assert!(has_node_edge(
        core,
        TemporalSceneColorLabel,
        Node3d::EndMainPass
    ));
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
