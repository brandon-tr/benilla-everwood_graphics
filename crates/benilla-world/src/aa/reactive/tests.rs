use super::*;

/// The CPU mirror of the shader's exact per-pixel comparison.
#[cfg(test)]
fn rgba_is_reactive(before: [f32; 4], after: [f32; 4]) -> bool {
    if !before
        .iter()
        .chain(after.iter())
        .all(|value| value.is_finite())
    {
        return true;
    }
    before != after
}

#[cfg(test)]
fn clamped_neighbor(pixel: UVec2, dx: i32, dy: i32, extent: UVec2) -> UVec2 {
    let x = (pixel.x as i32 + dx).clamp(0, extent.x.saturating_sub(1) as i32);
    let y = (pixel.y as i32 + dy).clamp(0, extent.y.saturating_sub(1) as i32);
    UVec2::new(x as u32, y as u32)
}

/// CPU mirror of the shader's clamped 3x3 dilation. The shader is the production implementation;
/// this helper exists to exercise exact detection/dilation behavior without GPU textures.
#[cfg(test)]
fn reactive_at(
    pixel: UVec2,
    extent: UVec2,
    sample: impl Fn(UVec2) -> ([f32; 4], [f32; 4]),
) -> bool {
    if extent == UVec2::ZERO {
        return false;
    }
    for dy in -1..=1 {
        for dx in -1..=1 {
            let neighbor = clamped_neighbor(pixel, dx, dy, extent);
            let (before, after) = sample(neighbor);
            if rgba_is_reactive(before, after) {
                return true;
            }
        }
    }
    false
}

fn key(main_view: u32, generation: u64, size: UVec2) -> DepthKey {
    DepthKey {
        main_view: Entity::from_raw_u32(main_view).unwrap(),
        generation,
        size,
    }
}

#[test]
fn identical_rgba_is_not_reactive() {
    let pixel = [0.25, 0.5, 0.75, 1.0];
    assert!(!rgba_is_reactive(pixel, pixel));
    assert!(!reactive_at(UVec2::new(2, 2), UVec2::new(5, 5), |_| {
        (pixel, pixel)
    }));
}

#[test]
fn any_changed_rgba_component_is_reactive() {
    let before = [0.1, 0.2, 0.3, 1.0];
    for component in 0..4 {
        let mut after = before;
        after[component] = 0.100_01;
        assert!(
            rgba_is_reactive(before, after),
            "component {component} change"
        );
    }
}

#[test]
fn nonfinite_before_or_after_is_reactive() {
    let finite = [0.1, 0.2, 0.3, 1.0];
    assert!(rgba_is_reactive([f32::NAN, 0.2, 0.3, 1.0], finite));
    assert!(rgba_is_reactive(finite, [0.1, f32::INFINITY, 0.3, 1.0]));
}

#[test]
fn changed_pixel_dilates_to_clamped_one_pixel_neighborhood() {
    let changed = UVec2::new(2, 2);
    let extent = UVec2::new(5, 5);
    let sample = |pixel: UVec2| {
        if pixel == changed {
            ([0.0, 0.0, 0.0, 1.0], [1.0, 0.0, 0.0, 1.0])
        } else {
            ([0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 1.0])
        }
    };
    assert!(reactive_at(changed, extent, sample));
    assert!(reactive_at(UVec2::new(1, 1), extent, sample));
    assert!(reactive_at(UVec2::new(3, 3), extent, sample));
    assert!(!reactive_at(UVec2::new(0, 0), extent, sample));
}

#[test]
fn unchanged_scene_does_not_become_reactive_at_borders() {
    let extent = UVec2::new(5, 5);
    let sample = |_| ([0.5, 0.5, 0.5, 1.0], [0.5, 0.5, 0.5, 1.0]);
    assert!(!reactive_at(UVec2::ZERO, extent, sample));
    assert!(!reactive_at(UVec2::new(4, 4), extent, sample));
    assert!(!reactive_at(UVec2::new(0, 4), extent, sample));
}

#[test]
fn snapshot_can_encode_without_prior_publication() {
    let generation = 7;
    let publication = DepthPublication::new(generation);
    assert!(!publication.is_current(1, generation));
    assert!(publication.encode(1, generation, || true));
    assert!(publication.is_current(1, generation));
}

#[test]
fn mask_requires_current_snapshot_and_does_not_require_prior_mask() {
    let generation = 7;
    let opaque = DepthPublication::new(generation);
    let mask = TemporalFramePair::new(generation, [10, 20]);
    let mut calls = 0;

    assert!(
        !encode_reactive_mask(&opaque, &mask, 1, generation, |slot| {
            calls += 1;
            assert_eq!(*slot, 20);
            true
        }),
        "missing snapshot must block the mask"
    );
    assert_eq!(calls, 0, "blocked mask must not encode a pass");
    assert!(mask.current(1, generation).is_none());
    assert!(mask.previous(1, generation).is_none());

    assert!(opaque.encode(1, generation, || true));
    assert!(
        encode_reactive_mask(&opaque, &mask, 1, generation, |slot| {
            calls += 1;
            assert_eq!(*slot, 20);
            true
        }),
        "first mask needs no prior mask publication"
    );
    assert_eq!(calls, 1);
    assert_eq!(mask.current(1, generation), Some(&20));
    assert!(mask.previous(1, generation).is_none());
    assert!(mask.current(2, generation).is_none());

    assert!(
        !encode_reactive_mask(&opaque, &mask, 2, generation, |slot| {
            calls += 1;
            assert_eq!(*slot, 10);
            true
        }),
        "stale snapshot cannot authorize a new epoch"
    );
    assert_eq!(calls, 1);
    assert!(mask.current(2, generation).is_none());
}

#[test]
fn mask_prerequisite_rejects_wrong_generation_zero_epoch_and_failed_encoder() {
    let generation = 7;
    let opaque = DepthPublication::new(generation);
    let mask = TemporalFramePair::new(generation, [10, 20]);
    assert!(opaque.encode(1, generation, || true));

    let mut calls = 0;
    assert!(
        !encode_reactive_mask(&opaque, &mask, 1, generation + 1, |_| {
            calls += 1;
            true
        }),
        "wrong generation blocks"
    );
    assert_eq!(calls, 0);
    assert!(
        !encode_reactive_mask(&opaque, &mask, 0, generation, |_| {
            calls += 1;
            true
        }),
        "epoch zero is invalid"
    );
    assert_eq!(calls, 0);

    assert!(
        !encode_reactive_mask(&opaque, &mask, 1, generation, |slot| {
            calls += 1;
            assert_eq!(*slot, 20);
            false
        }),
        "failed encoder must not publish"
    );
    assert_eq!(calls, 1);
    assert!(mask.current(1, generation).is_none());
    assert!(mask.previous(1, generation).is_none());
}

#[test]
fn skipped_snapshot_or_mask_does_not_publish_current_epoch() {
    let generation = 7;
    let opaque = DepthPublication::new(generation);
    let mask = TemporalFramePair::new(generation, [10, 20]);
    opaque.encode(1, generation, || true);
    assert!(!opaque.is_current(2, generation));
    assert!(!encode_reactive_mask(&opaque, &mask, 2, generation, |_| {
        false
    }));
    assert!(mask.current(2, generation).is_none());
    assert!(mask.previous(2, generation).is_none());
    assert!(opaque.is_current(1, generation));
}

#[test]
fn consecutive_reactive_masks_expose_the_prior_payload() {
    let generation = 7;
    let opaque = DepthPublication::new(generation);
    let mask = TemporalFramePair::new(generation, [10, 20]);

    assert!(opaque.encode(1, generation, || true));
    assert!(encode_reactive_mask(
        &opaque,
        &mask,
        1,
        generation,
        |slot| {
            assert_eq!(*slot, 20);
            true
        }
    ));
    assert_eq!(mask.current(1, generation), Some(&20));
    assert!(mask.previous(1, generation).is_none());

    assert!(opaque.encode(2, generation, || true));
    assert!(encode_reactive_mask(
        &opaque,
        &mask,
        2,
        generation,
        |slot| {
            assert_eq!(*slot, 10);
            true
        }
    ));
    assert_eq!(mask.current(2, generation), Some(&10));
    assert_eq!(mask.previous(2, generation), Some(&20));
}

#[test]
fn view_generation_extent_and_disable_retire_reactive_state() {
    let requested = key(1, 1, UVec2::new(800, 600));
    assert_eq!(
        reactive_decision(Some(requested), requested, true),
        ReactiveDecision::Keep
    );
    assert_eq!(
        reactive_decision(Some(key(2, 1, requested.size)), requested, true),
        ReactiveDecision::Replace
    );
    assert_eq!(
        reactive_decision(Some(key(1, 2, requested.size)), requested, true),
        ReactiveDecision::Replace
    );
    assert_eq!(
        reactive_decision(Some(key(1, 1, UVec2::new(640, 480))), requested, true),
        ReactiveDecision::Replace
    );
    assert_eq!(
        reactive_decision(Some(requested), requested, false),
        ReactiveDecision::Remove
    );
    assert_eq!(
        reactive_decision(None, requested, true),
        ReactiveDecision::Replace
    );
}

#[test]
fn fallback_requests_no_reactive_allocations() {
    let requests = depth::TemporalDepthRequests::default();
    assert!(requests.iter().next().is_none());
    assert_eq!(
        reactive_decision(None, key(1, 1, UVec2::new(800, 600)), false),
        ReactiveDecision::Remove
    );
}

#[test]
fn engine_registration_does_not_require_app_ao_or_water_labels() {
    let production = compact_production(include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/src/aa/reactive.rs"
    )));
    assert!(!production.contains("AoLabel"));
    assert!(!production.contains("WaterDepthLabel"));

    let mut app = App::new();
    app.insert_resource(crate::view::MsaaSetting { samples: 1 })
        .init_resource::<crate::view::Viewer>()
        .init_resource::<crate::schedule::WorldLive>()
        .add_message::<crate::world_map::MapChange>()
        .add_plugins(super::super::AaPlugin);
    app.update();
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

const SNAPSHOT_EDGE: &str =
    "Node3d::MainOpaquePass,TemporalOpaqueSnapshotLabel,Node3d::MainTransmissivePass,";
const REACTIVE_EDGE: &str =
    "Node3d::MainTransparentPass,TemporalReactiveLabel,Node3d::EndMainPass,";
const AO_EDGE: &str = "Node3d::MainOpaquePass,AoLabel,TemporalOpaqueSnapshotLabel,WaterDepthLabel,";
const WATER_EDGE: &str = "Node3d::MainOpaquePass,WaterDepthLabel,Node3d::MainTransparentPass,";

/// Production text only, with line comments removed. Test modules can never satisfy a graph
/// assertion, and a comment cannot masquerade as registration evidence.
fn compact_production(source: &str) -> String {
    let production = source.split("#[cfg(test)]").next().unwrap_or(source);
    let without_comments: String = production
        .lines()
        .map(|line| line.split("//").next().unwrap_or(""))
        .collect();
    without_comments.split_whitespace().collect()
}

fn graph_sequences_are_present(engine: &str, ssao: &str, water: &str) -> bool {
    let engine = compact_production(engine);
    let ssao = compact_production(ssao);
    let water = compact_production(water);
    engine.contains(SNAPSHOT_EDGE)
        && engine.contains(REACTIVE_EDGE)
        && ssao.contains(AO_EDGE)
        && water.contains(WATER_EDGE)
}

#[test]
fn graph_order_places_snapshot_and_mask_at_required_boundaries() {
    let engine = include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/src/aa/reactive.rs"));
    let ssao = include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../benilla-app/src/ssao.rs"
    ));
    let water = include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/src/liquid/scene_depth.rs"
    ));
    assert!(graph_sequences_are_present(engine, ssao, water));

    // Mutation regression: each production edge is independently required.
    let production = compact_production(engine);
    let without_snapshot = production.replace(SNAPSHOT_EDGE, "");
    assert!(!graph_sequences_are_present(&without_snapshot, ssao, water));
    let without_reactive = production.replace(REACTIVE_EDGE, "");
    assert!(!graph_sequences_are_present(&without_reactive, ssao, water));

    // Expected text in a test section must not rescue a missing production edge.
    let synthetic = format!(
        "{without_snapshot}\n#[cfg(test)]\nmod tests {{\n\
             const SNAPSHOT: &str = \"{SNAPSHOT_EDGE}\";\n\
             const REACTIVE: &str = \"{REACTIVE_EDGE}\";\n\
             assert!(true);\n}}"
    );
    assert!(!graph_sequences_are_present(&synthetic, ssao, water));
}
