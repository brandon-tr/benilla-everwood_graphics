use super::*;

fn entity(id: u32) -> Entity {
    Entity::from_raw_u32(id).expect("test entity")
}

fn history_for(main_view: Entity, size: UVec2) -> TemporalHistory {
    let mut history = TemporalHistory::default();
    history.register(main_view, TemporalTarget::new(size, 1.0));
    history
}

fn camera(active: bool) -> Camera {
    Camera {
        is_active: active,
        ..default()
    }
}

/// A requested TAA that fell back to MSAA must not create a request.
#[test]
fn fallback_requests_no_temporal_depth() {
    let (effective, reason) = AaPolicy::resolve(
        super::super::AaMode::Taa,
        4,
        super::super::AaReadiness::default(),
    );
    assert_eq!(effective, AaEffective::Msaa { samples: 4 });
    assert_eq!(reason, super::super::AaReason::TaaUnavailable);
    let main_view = entity(1);
    let history = history_for(main_view, UVec2::new(800, 600));
    assert!(
        eligible_request(
            main_view,
            effective,
            &camera(true),
            Msaa::Sample4,
            Some(TemporalTarget::new(UVec2::new(800, 600), 1.0)),
            history.view(main_view),
        )
        .is_none(),
        "an MSAA fallback must not allocate or draw temporal depth"
    );
}

/// Only an active, single-sample, full-target world view with a registered history is eligible.
#[test]
fn depth_eligibility_excludes_auxiliary_inactive_and_multisampled_views() {
    let main_view = entity(2);
    let target = TemporalTarget::new(UVec2::new(800, 600), 1.0);
    let history = history_for(main_view, UVec2::new(800, 600));
    let eligible = |camera: &Camera, msaa: Msaa| {
        eligible_request(
            main_view,
            AaEffective::Taa,
            camera,
            msaa,
            Some(target),
            history.view(main_view),
        )
        .is_some()
    };

    assert!(eligible(&camera(true), Msaa::Off));
    assert!(!eligible(&camera(false), Msaa::Off));
    assert!(!eligible(&camera(true), Msaa::Sample4));

    let mut custom_viewport = camera(true);
    custom_viewport.viewport = Some(bevy::camera::Viewport {
        physical_position: UVec2::ZERO,
        physical_size: UVec2::new(400, 300),
        depth: 0.0..1.0,
    });
    assert!(!eligible(&custom_viewport, Msaa::Off));

    let mut sub_view = camera(true);
    sub_view.sub_camera_view = Some(bevy::camera::SubCameraView {
        full_size: UVec2::new(800, 600),
        offset: Vec2::ZERO,
        size: UVec2::new(400, 300),
    });
    assert!(!eligible(&sub_view, Msaa::Off));

    assert!(
        eligible_request(
            main_view,
            AaEffective::Msaa { samples: 1 },
            &camera(true),
            Msaa::Off,
            Some(target),
            history.view(main_view),
        )
        .is_none(),
        "MSAA effective is not a temporal request"
    );
    assert!(
        eligible_request(
            main_view,
            AaEffective::Taa,
            &camera(true),
            Msaa::Off,
            Some(target),
            None,
        )
        .is_none(),
        "a view without registered history has no generation to publish"
    );
}

/// Requests are keyed by render entity, carry the distinct main identity, and a new extraction
/// clears a view that vanished.
#[test]
fn extracted_requests_use_render_identity_and_refresh_epoch() {
    let mut requests = TemporalDepthRequests::default();
    let main_view = entity(3);
    let render_view = entity(4);
    let request = TemporalDepthRequest {
        main_view,
        generation: 7,
        size: UVec2::new(800, 600),
    };
    let epoch = requests.begin_epoch();
    assert_eq!(epoch, 1);
    requests.insert(render_view, request);
    assert_eq!(requests.get(render_view), Some(&request));
    assert_eq!(requests.get(main_view), None, "main and render ids differ");
    assert_eq!(requests.get(render_view).unwrap().main_view, main_view);

    let next = requests.begin_epoch();
    assert_eq!(next, 2);
    assert!(
        requests.get(render_view).is_none(),
        "a view absent from the new extraction must not leave a stale request"
    );
    requests.insert(render_view, request);
    assert!(requests.contains(render_view));
}

/// Allocation reuse is exactly key equality with a valid source.
#[test]
fn allocation_plan_reuses_only_matching_view_generation_and_extent() {
    let key = DepthKey {
        main_view: entity(5),
        generation: 2,
        size: UVec2::new(800, 600),
    };
    assert_eq!(
        allocation_decision(None, key, true),
        AllocationDecision::Replace
    );
    assert_eq!(
        allocation_decision(Some(key), key, true),
        AllocationDecision::Keep
    );
    for changed in [
        DepthKey {
            main_view: entity(6),
            ..key
        },
        DepthKey {
            generation: 3,
            ..key
        },
        DepthKey {
            size: UVec2::new(640, 480),
            ..key
        },
    ] {
        assert_eq!(
            allocation_decision(Some(key), changed, true),
            AllocationDecision::Replace
        );
    }
    assert_eq!(
        allocation_decision(Some(key), key, false),
        AllocationDecision::Remove,
        "an invalid source retires even a matching allocation"
    );
}

/// Publication is current-epoch and current-generation only; a skipped encode leaves the prior
/// stamp stale.
#[test]
fn depth_publication_requires_current_epoch_and_generation() {
    let publication = DepthPublication::new(4);
    assert!(
        !publication.is_current(0, 4),
        "fresh allocation is not current"
    );
    assert!(!publication.is_current(1, 4), "no encode has happened");

    publication.publish(1);
    assert!(publication.is_current(1, 4));
    assert!(!publication.is_current(0, 4), "previous epoch is stale");
    assert!(
        !publication.is_current(2, 4),
        "future/next epoch is not current"
    );
    assert!(
        !publication.is_current(1, 5),
        "another generation cannot reuse this stamp"
    );

    // A skipped encode leaves the previous epoch behind, so the current extraction is rejected.
    let skipped = DepthPublication::new(4);
    skipped.publish(2);
    assert!(!skipped.is_current(3, 4));
}

/// A fresh allocation has no published content, but the producer must still be able to encode
/// its first frame; otherwise it can never publish anything.
#[test]
fn producer_can_encode_an_unpublished_allocation() {
    let publication = DepthPublication::new(4);
    assert!(!publication.is_current(1, 4), "fresh allocation");
    let mut calls = 0;
    let encoded = publication.encode(1, 4, || {
        calls += 1;
        true
    });
    assert!(encoded, "the first frame must encode");
    assert_eq!(calls, 1, "the encode closure runs exactly once");
    assert!(publication.is_current(1, 4));
    assert!(!publication.is_current(2, 4));
}

/// Once epoch 1 is published, epoch 2 is not current yet, but the producer must still be able
/// to encode it and make the old epoch stale.
#[test]
fn producer_can_encode_the_next_epoch() {
    let publication = DepthPublication::new(4);
    publication.publish(1);
    assert!(publication.is_current(1, 4));
    let mut calls = 0;
    let encoded = publication.encode(2, 4, || {
        calls += 1;
        true
    });
    assert!(encoded, "the next frame must encode");
    assert_eq!(calls, 1);
    assert!(publication.is_current(2, 4));
    assert!(!publication.is_current(1, 4), "epoch 1 is stale");
}

/// A failed or skipped encode publishes nothing; a previous epoch remains unavailable for the
/// current extraction.
#[test]
fn failed_or_skipped_encoding_does_not_publish_current_epoch() {
    let publication = DepthPublication::new(4);
    let mut calls = 0;
    assert!(!publication.encode(1, 4, || {
        calls += 1;
        false
    }));
    assert_eq!(calls, 1, "a failed encode still ran once");
    assert!(!publication.is_current(1, 4));

    publication.publish(1);
    assert!(!publication.encode(2, 4, || {
        calls += 1;
        false
    }));
    assert_eq!(calls, 2);
    assert!(!publication.is_current(2, 4));
    assert!(
        publication.is_current(1, 4),
        "the prior epoch remains, but is not current for extraction 2"
    );
}

/// A zero epoch or another generation never runs the closure and never publishes.
#[test]
fn producer_rejects_zero_epoch_and_obsolete_generation() {
    let publication = DepthPublication::new(4);
    let mut calls = 0;
    assert!(!publication.encode(0, 4, || {
        calls += 1;
        true
    }));
    assert!(!publication.encode(1, 5, || {
        calls += 1;
        true
    }));
    assert_eq!(
        calls, 0,
        "rejected encodes must not touch the command stream"
    );
    assert!(!publication.is_current(1, 4));
}

/// The copy target is native, full-target, reverse-depth-preserving and single-sample.
#[test]
fn temporal_depth_descriptor_is_native_single_sample_r32float() {
    let descriptor = depth_descriptor(UVec2::new(800, 600));
    assert_eq!(descriptor.size.width, 800);
    assert_eq!(descriptor.size.height, 600);
    assert_eq!(descriptor.size.depth_or_array_layers, 1);
    assert_eq!(descriptor.mip_level_count, 1);
    assert_eq!(descriptor.sample_count, 1);
    assert_eq!(descriptor.dimension, TextureDimension::D2);
    assert_eq!(descriptor.format, TextureFormat::R32Float);
    assert!(descriptor
        .usage
        .contains(TextureUsages::RENDER_ATTACHMENT | TextureUsages::TEXTURE_BINDING));
}

#[test]
fn first_depth_capture_has_no_previous() {
    let frames = TemporalFramePair::new(4, [10, 20]);
    assert_eq!(frames.current(1, 4), None);
    assert_eq!(frames.previous(1, 4), None);
    assert!(frames.encode_current(1, 4, |slot| {
        assert_eq!(*slot, 20);
        true
    }));
    assert_eq!(frames.current(1, 4), Some(&20));
    assert_eq!(frames.previous(1, 4), None);
}

#[test]
fn consecutive_depth_captures_keep_previous() {
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
    assert_ne!(frames.current(3, 4), frames.previous(3, 4));
}

#[test]
fn missing_depth_capture_breaks_previous() {
    let frames = TemporalFramePair::new(4, [10, 20]);
    assert!(frames.encode_current(1, 4, |_| true));
    assert!(!frames.encode_current(2, 4, |_| false));
    assert!(frames.encode_current(3, 4, |_| true));
    assert_eq!(frames.current(3, 4), Some(&20));
    assert_eq!(frames.previous(3, 4), None);
    assert!(frames.encode_current(4, 4, |_| true));
    assert_eq!(frames.current(4, 4), Some(&10));
    assert_eq!(frames.previous(4, 4), Some(&20));
}

#[test]
fn replacing_depth_key_starts_without_previous() {
    let key = DepthKey {
        main_view: Entity::from_raw_u32(1).unwrap(),
        generation: 4,
        size: UVec2::new(800, 600),
    };
    let replacement = DepthKey {
        generation: 5,
        ..key
    };
    assert_eq!(
        allocation_decision(Some(key), replacement, true),
        AllocationDecision::Replace
    );
    let old = TemporalFramePair::new(key.generation, [10, 20]);
    assert!(old.encode_current(1, key.generation, |_| true));
    assert_eq!(old.current(1, key.generation), Some(&20));

    let fresh = TemporalFramePair::new(replacement.generation, [10, 20]);
    assert_eq!(fresh.current(1, replacement.generation), None);
    assert_eq!(fresh.previous(1, replacement.generation), None);
    assert_eq!(fresh.current(1, key.generation), None);
}

#[test]
fn depth_pair_rejects_zero_and_wrong_generation_without_encoding() {
    let frames = TemporalFramePair::new(4, [10, 20]);
    let calls = std::cell::Cell::new(0);
    assert!(!frames.encode_current(0, 4, |_| {
        calls.set(calls.get() + 1);
        true
    }));
    assert!(!frames.encode_current(1, 5, |_| {
        calls.set(calls.get() + 1);
        true
    }));
    assert_eq!(calls.get(), 0);
    assert_eq!(frames.current(1, 4), None);
    assert_eq!(frames.previous(1, 4), None);
}

/// The child plugin must not require a render app or GPU to run the existing headless AA tests.
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
