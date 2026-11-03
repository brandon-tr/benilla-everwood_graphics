use super::*;

fn view(app: &mut App) -> Entity {
    app.world_mut().spawn_empty().id()
}

fn commit(app: &mut App, view: Entity) -> HistoryToken {
    let mut history = app.world_mut().resource_mut::<TemporalHistory>();
    let token = history.begin(view).expect("registered view");
    assert!(history.commit(token), "the current token must commit");
    token
}

/// The default request is the existing MSAA path, and unfinished temporal modes never resolve
/// to a temporal effective mode.
#[test]
fn the_default_is_msaa_and_unfinished_modes_fall_back_with_a_reason() {
    let ready_none = AaReadiness::default();
    assert_eq!(
        AaPolicy::resolve(AaMode::Msaa, 4, ready_none),
        (AaEffective::Msaa { samples: 4 }, AaReason::MsaaRequested)
    );
    assert_eq!(
        AaPolicy::resolve(AaMode::Taa, 4, ready_none),
        (AaEffective::Msaa { samples: 4 }, AaReason::TaaUnavailable)
    );
    assert_eq!(
        AaPolicy::resolve(AaMode::TemporalUpscale, 2, ready_none),
        (
            AaEffective::Msaa { samples: 2 },
            AaReason::TemporalUpscaleUnavailable
        )
    );
    assert_eq!(
        AaPolicy::resolve(AaMode::Off, 1, ready_none),
        (AaEffective::Off, AaReason::OffRequested)
    );
    assert_eq!(
        AaPolicy::resolve(AaMode::Off, 8, ready_none),
        (AaEffective::Msaa { samples: 8 }, AaReason::OffUnavailable)
    );
    let default = AaPolicy::default();
    assert_eq!(default.requested(), AaMode::Msaa);
    assert_eq!(default.effective(), AaEffective::Msaa { samples: 1 });
    assert!(!default.effective().is_temporal());
}

/// A request becomes effective only once its implementation readiness is true. The app's boot
/// selection sets the same flag through `AaReadiness::set_native_taa`.
#[test]
fn readiness_is_the_only_way_a_temporal_mode_becomes_effective() {
    let ready = AaReadiness {
        taa: true,
        temporal_upscale: false,
    };
    assert_eq!(
        AaPolicy::resolve(AaMode::Taa, 1, ready),
        (AaEffective::Taa, AaReason::TaaRequested)
    );
    assert_eq!(
        AaPolicy::resolve(AaMode::TemporalUpscale, 1, ready),
        (
            AaEffective::Msaa { samples: 1 },
            AaReason::TemporalUpscaleUnavailable
        )
    );
}

/// A registered view starts invalid; only a commit makes its history sampleable.
#[test]
fn first_frame_history_is_invalid_until_committed() {
    let mut app = App::new();
    app.init_resource::<TemporalHistory>();
    let view = view(&mut app);
    app.world_mut()
        .resource_mut::<TemporalHistory>()
        .register(view, TemporalTarget::default());
    let mut history = app.world_mut().resource_mut::<TemporalHistory>();
    let token = history.begin(view).expect("registered");
    assert!(!token.had_history(), "first frame has no previous pair");
    assert!(!history.is_valid(view));
    assert!(history.commit(token));
    assert!(history.is_valid(view));
    assert_eq!(history.view(view).unwrap().last_commit_frame(), Some(1));
}

/// A frame that is abandoned, or begun and never finished, must not leave a previous pair
/// sampleable; the next begin reports no history.
#[test]
fn a_skipped_frame_does_not_mark_history_valid() {
    let mut app = App::new();
    app.init_resource::<TemporalHistory>();
    let view = view(&mut app);
    let mut history = app.world_mut().resource_mut::<TemporalHistory>();
    history.register(view, TemporalTarget::default());

    let token = history.begin(view).expect("registered");
    assert!(history.abandon(token));
    assert!(!history.is_valid(view));
    let next = history.begin(view).expect("registered");
    assert!(
        !next.had_history(),
        "an abandoned frame leaves nothing to sample"
    );
    assert!(history.commit(next));

    // A begin with no commit/abandon is a skipped frame too, and invalidates the valid pair.
    let _uncommitted = history.begin(view).expect("registered");
    let after = history.begin(view).expect("registered");
    assert!(
        !after.had_history(),
        "an uncommitted frame must not leave the old pair reusable"
    );
    assert_eq!(
        history.view(view).unwrap().last_invalidation(),
        AaInvalidation::FrameNotCommitted
    );
}

/// Work from before an invalidation cannot commit into the new generation.
#[test]
fn an_obsolete_generation_commit_is_rejected() {
    let mut app = App::new();
    app.init_resource::<TemporalHistory>();
    let view = view(&mut app);
    let mut history = app.world_mut().resource_mut::<TemporalHistory>();
    history.register(view, TemporalTarget::default());
    let stale = history.begin(view).expect("registered");
    history.invalidate(view, AaInvalidation::CameraCut);
    assert!(
        !history.commit(stale),
        "a commit from the invalidated generation must fail"
    );
    assert!(!history.is_valid(view));
    let fresh = history.begin(view).expect("registered");
    assert_eq!(fresh.generation(), stale.generation() + 1);
    assert!(history.commit(fresh));
    assert!(!history.commit(stale), "a stale token stays stale");
}

/// Two views keep independent histories: one invalidation or commit never crosses over.
#[test]
fn independent_views_keep_independent_lifetimes() {
    let mut app = App::new();
    app.init_resource::<TemporalHistory>();
    let (a, b) = (view(&mut app), view(&mut app));
    let mut history = app.world_mut().resource_mut::<TemporalHistory>();
    history.register(a, TemporalTarget::default());
    history.register(b, TemporalTarget::default());
    let token_a = history.begin(a).expect("a");
    let token_b = history.begin(b).expect("b");
    assert!(history.commit(token_a));
    assert!(history.commit(token_b));
    history.invalidate(b, AaInvalidation::Teleport);
    assert!(history.is_valid(a));
    assert!(!history.is_valid(b));
    assert_eq!(history.view(a).unwrap().generation(), 1);
    assert_eq!(history.view(b).unwrap().generation(), 2);
    assert!(history.unregister(b));
    assert!(history.is_valid(a));
    assert!(!history.is_registered(b));
}

/// A target size or render-scale change invalidates only the changed view.
#[test]
fn a_target_change_invalidates_that_view() {
    let mut app = App::new();
    app.init_resource::<TemporalHistory>();
    let (a, b) = (view(&mut app), view(&mut app));
    let mut history = app.world_mut().resource_mut::<TemporalHistory>();
    history.register(a, TemporalTarget::new(UVec2::new(800, 600), 1.0));
    history.register(b, TemporalTarget::new(UVec2::new(800, 600), 1.0));
    let token_a = history.begin(a).expect("a");
    let token_b = history.begin(b).expect("b");
    assert!(history.commit(token_a));
    assert!(history.commit(token_b));
    history.set_target(a, TemporalTarget::new(UVec2::new(800, 600), 2.0));
    assert!(!history.is_valid(a));
    assert_eq!(
        history.view(a).unwrap().last_invalidation(),
        AaInvalidation::TargetChanged
    );
    assert!(history.is_valid(b), "the other view's target did not move");
    history.set_target(b, TemporalTarget::new(UVec2::new(800, 600), 1.0));
    assert!(history.is_valid(b), "an unchanged target is a no-op");
}

/// The plugin registers `WorldCamera` only, and a despawn drops its slot.
#[test]
fn the_plugin_registers_and_unregisters_world_views() {
    let mut app = test_app();
    let view = app
        .world_mut()
        .spawn((WorldCamera, Camera::default(), Msaa::Off))
        .id();
    app.update();
    assert!(app
        .world()
        .resource::<TemporalHistory>()
        .is_registered(view));
    app.world_mut().despawn(view);
    app.update();
    assert!(!app
        .world()
        .resource::<TemporalHistory>()
        .is_registered(view));
}

/// The lifecycle hooks the plugin owns: projection/target/camera activity, world-live, cover
/// and map changes all invalidate the view's committed history.
#[test]
fn available_lifecycle_signals_invalidate_committed_history() {
    let mut app = test_app();
    let view = app
        .world_mut()
        .spawn((
            WorldCamera,
            Camera::default(),
            Msaa::Off,
            Projection::from(PerspectiveProjection::default()),
            TemporalTarget::default(),
        ))
        .id();
    app.update();
    commit(&mut app, view);

    // A projection change.
    app.world_mut()
        .get_mut::<Projection>(view)
        .unwrap()
        .clone_from(&Projection::from(PerspectiveProjection {
            fov: 0.9,
            ..default()
        }));
    app.update();
    assert!(!app.world().resource::<TemporalHistory>().is_valid(view));
    assert_eq!(
        app.world()
            .resource::<TemporalHistory>()
            .view(view)
            .unwrap()
            .last_invalidation(),
        AaInvalidation::ProjectionChanged
    );

    // A target change.
    commit(&mut app, view);
    *app.world_mut().get_mut::<TemporalTarget>(view).unwrap() =
        TemporalTarget::new(UVec2::new(640, 480), 1.5);
    app.update();
    assert!(!app.world().resource::<TemporalHistory>().is_valid(view));
    assert_eq!(
        app.world()
            .resource::<TemporalHistory>()
            .view(view)
            .unwrap()
            .last_invalidation(),
        AaInvalidation::TargetChanged
    );

    // The loading cover.
    commit(&mut app, view);
    app.world_mut().resource_mut::<Viewer>().world_covered = true;
    app.update();
    assert!(!app.world().resource::<TemporalHistory>().is_valid(view));
    assert_eq!(
        app.world()
            .resource::<TemporalHistory>()
            .view(view)
            .unwrap()
            .last_invalidation(),
        AaInvalidation::LoadingCover
    );

    // A cross-map transition.
    commit(&mut app, view);
    app.world_mut().write_message(MapChange);
    app.update();
    assert!(!app.world().resource::<TemporalHistory>().is_valid(view));
    assert_eq!(
        app.world()
            .resource::<TemporalHistory>()
            .view(view)
            .unwrap()
            .last_invalidation(),
        AaInvalidation::WorldTransition
    );

    // Leaving the world.
    commit(&mut app, view);
    app.world_mut().resource_mut::<WorldLive>().0 = false;
    app.update();
    assert!(!app.world().resource::<TemporalHistory>().is_valid(view));
    assert_eq!(
        app.world()
            .resource::<TemporalHistory>()
            .view(view)
            .unwrap()
            .last_invalidation(),
        AaInvalidation::WorldLeft
    );
}

/// An effective AA-mode change invalidates history; a request that falls back to the same
/// effective path does not.
#[test]
fn an_effective_mode_change_invalidates_history() {
    let mut app = test_app();
    let view = app
        .world_mut()
        .spawn((WorldCamera, Camera::default(), Msaa::Off))
        .id();
    app.update();
    commit(&mut app, view);
    {
        let mut policy = app.world_mut().resource_mut::<AaPolicy>();
        policy.request(AaMode::Taa);
    }
    app.update();
    assert_eq!(
        app.world().resource::<AaPolicy>().effective(),
        AaEffective::Msaa { samples: 1 }
    );
    assert_eq!(
        app.world().resource::<AaPolicy>().reason(),
        AaReason::TaaUnavailable
    );
    assert!(
        app.world().resource::<TemporalHistory>().is_valid(view),
        "a fallback that keeps the same effective path is not a mode change"
    );

    app.world_mut()
        .resource_mut::<AaPolicy>()
        .request(AaMode::Off);
    app.update();
    assert_eq!(
        app.world().resource::<AaPolicy>().effective(),
        AaEffective::Off
    );
    assert!(!app.world().resource::<TemporalHistory>().is_valid(view));
    assert_eq!(
        app.world()
            .resource::<TemporalHistory>()
            .view(view)
            .unwrap()
            .last_invalidation(),
        AaInvalidation::AaModeChanged
    );
}

/// A writer may take `&mut Projection`/`&mut Camera` and write the same values every frame
/// (`apply_scoped_view` does exactly that). Change ticks are candidates, not causes: committed
/// history and its generation must survive equal writes.
#[test]
fn unchanged_projection_and_camera_writes_preserve_history() {
    let mut app = test_app();
    let view = app
        .world_mut()
        .spawn((
            WorldCamera,
            Camera::default(),
            Msaa::Off,
            Projection::from(PerspectiveProjection::default()),
        ))
        .id();
    app.update();
    commit(&mut app, view);
    let generation = app
        .world()
        .resource::<TemporalHistory>()
        .view(view)
        .unwrap()
        .generation();

    for frame in 0..3 {
        // Equal mutable writes, exactly the kind the scoped-view writer makes.
        {
            let mut projection = app.world_mut().get_mut::<Projection>(view).unwrap();
            let same = projection.clone();
            *projection = same;
        }
        {
            let mut camera = app.world_mut().get_mut::<Camera>(view).unwrap();
            let same = camera.clone();
            *camera = same;
        }
        app.update();
        let history = app.world().resource::<TemporalHistory>();
        assert!(
            history.is_valid(view),
            "frame {frame}: an equal rewrite must preserve committed history"
        );
        assert_eq!(
            history.view(view).unwrap().generation(),
            generation,
            "frame {frame}: an equal rewrite must not bump the generation"
        );
    }
}

/// Real projection values invalidate with `ProjectionChanged`; equal rewrites after a commit
/// keep the pair.
#[test]
fn meaningful_projection_changes_invalidate_history() {
    let mut app = test_app();
    let view = app
        .world_mut()
        .spawn((
            WorldCamera,
            Camera::default(),
            Msaa::Off,
            Projection::from(PerspectiveProjection::default()),
        ))
        .id();
    app.update();
    commit(&mut app, view);

    let mutate = |app: &mut App, f: &dyn Fn(&mut PerspectiveProjection)| {
        let mut projection = app.world_mut().get_mut::<Projection>(view).unwrap();
        let Projection::Perspective(perspective) = &mut *projection else {
            panic!("spawned perspective");
        };
        f(perspective);
    };

    mutate(&mut app, &|p| p.fov = 0.9);
    app.update();
    assert!(!app.world().resource::<TemporalHistory>().is_valid(view));
    assert_eq!(
        app.world()
            .resource::<TemporalHistory>()
            .view(view)
            .unwrap()
            .last_invalidation(),
        AaInvalidation::ProjectionChanged
    );

    commit(&mut app, view);
    mutate(&mut app, &|p| p.aspect_ratio = 1.5);
    app.update();
    assert!(!app.world().resource::<TemporalHistory>().is_valid(view));
    assert_eq!(
        app.world()
            .resource::<TemporalHistory>()
            .view(view)
            .unwrap()
            .last_invalidation(),
        AaInvalidation::ProjectionChanged
    );

    commit(&mut app, view);
    // `near_clip_plane` carries the same plane; move both so this is a real near change rather
    // than a field the matrix ignores.
    mutate(&mut app, &|p| {
        p.near = 0.2;
        p.near_clip_plane = Vec4::new(0.0, 0.0, -1.0, -0.2);
    });
    app.update();
    assert!(!app.world().resource::<TemporalHistory>().is_valid(view));
    assert_eq!(
        app.world()
            .resource::<TemporalHistory>()
            .view(view)
            .unwrap()
            .last_invalidation(),
        AaInvalidation::ProjectionChanged
    );

    // An equal rewrite after the last commit preserves the pair.
    commit(&mut app, view);
    {
        let mut projection = app.world_mut().get_mut::<Projection>(view).unwrap();
        let same = projection.clone();
        *projection = same;
    }
    app.update();
    assert!(app.world().resource::<TemporalHistory>().is_valid(view));
}

/// Real camera values invalidate: activity off/on, viewport position/size/depth and the
/// sub-camera view each report their documented reason.
#[test]
fn meaningful_camera_changes_invalidate_history() {
    let mut app = test_app();
    let view = app
        .world_mut()
        .spawn((
            WorldCamera,
            Camera::default(),
            Msaa::Off,
            Projection::from(PerspectiveProjection::default()),
        ))
        .id();
    app.update();
    commit(&mut app, view);

    // Disabling and re-enabling have distinct reasons.
    app.world_mut().get_mut::<Camera>(view).unwrap().is_active = false;
    app.update();
    assert_eq!(
        app.world()
            .resource::<TemporalHistory>()
            .view(view)
            .unwrap()
            .last_invalidation(),
        AaInvalidation::ViewInactive
    );
    commit(&mut app, view);
    app.world_mut().get_mut::<Camera>(view).unwrap().is_active = true;
    app.update();
    assert_eq!(
        app.world()
            .resource::<TemporalHistory>()
            .view(view)
            .unwrap()
            .last_invalidation(),
        AaInvalidation::RenderingInterrupted
    );

    // A viewport appearing, then each explicit field changing.
    let viewport =
        |position: UVec2, size: UVec2, depth: std::ops::Range<f32>| bevy::camera::Viewport {
            physical_position: position,
            physical_size: size,
            depth,
        };
    commit(&mut app, view);
    app.world_mut().get_mut::<Camera>(view).unwrap().viewport =
        Some(viewport(UVec2::ZERO, UVec2::new(640, 480), 0.0..1.0));
    app.update();
    assert_eq!(
        app.world()
            .resource::<TemporalHistory>()
            .view(view)
            .unwrap()
            .last_invalidation(),
        AaInvalidation::ViewportChanged
    );

    commit(&mut app, view);
    app.world_mut().get_mut::<Camera>(view).unwrap().viewport =
        Some(viewport(UVec2::new(1, 2), UVec2::new(640, 480), 0.0..1.0));
    app.update();
    assert_eq!(
        app.world()
            .resource::<TemporalHistory>()
            .view(view)
            .unwrap()
            .last_invalidation(),
        AaInvalidation::ViewportChanged
    );

    commit(&mut app, view);
    app.world_mut().get_mut::<Camera>(view).unwrap().viewport =
        Some(viewport(UVec2::new(1, 2), UVec2::new(800, 600), 0.0..1.0));
    app.update();
    assert_eq!(
        app.world()
            .resource::<TemporalHistory>()
            .view(view)
            .unwrap()
            .last_invalidation(),
        AaInvalidation::ViewportChanged
    );

    commit(&mut app, view);
    app.world_mut().get_mut::<Camera>(view).unwrap().viewport =
        Some(viewport(UVec2::new(1, 2), UVec2::new(800, 600), 0.25..0.75));
    app.update();
    assert_eq!(
        app.world()
            .resource::<TemporalHistory>()
            .view(view)
            .unwrap()
            .last_invalidation(),
        AaInvalidation::ViewportChanged
    );

    // A sub-camera view is a distinct explicit value too.
    commit(&mut app, view);
    app.world_mut()
        .get_mut::<Camera>(view)
        .unwrap()
        .sub_camera_view = Some(SubCameraView {
        full_size: UVec2::new(800, 600),
        offset: Vec2::ZERO,
        size: UVec2::new(400, 300),
    });
    app.update();
    assert_eq!(
        app.world()
            .resource::<TemporalHistory>()
            .view(view)
            .unwrap()
            .last_invalidation(),
        AaInvalidation::ViewportChanged
    );
}

/// Re-registering a view starts with no snapshots from its prior lifetime.
#[test]
fn a_reused_view_starts_without_old_snapshots() {
    let mut app = test_app();
    let view = app
        .world_mut()
        .spawn((
            WorldCamera,
            Camera::default(),
            Msaa::Off,
            Projection::from(PerspectiveProjection::default()),
        ))
        .id();
    app.update();
    commit(&mut app, view);
    {
        let mut history = app.world_mut().resource_mut::<TemporalHistory>();
        assert!(history.unregister(view));
        history.register(view, TemporalTarget::default());
        let slot = history.view(view).unwrap();
        assert!(
            slot.projection.is_none(),
            "a re-registered view must not inherit the prior projection snapshot"
        );
        assert!(
            slot.camera.is_none(),
            "a re-registered view must not inherit the prior camera snapshot"
        );
    }
}

/// An Off request cannot be advertised while the camera still renders multisampled: the
/// effective mode stays the actual MSAA path and names the reason.
#[test]
fn off_request_reports_actual_multisampling() {
    let mut app = test_app();
    let view = app
        .world_mut()
        .spawn((
            WorldCamera,
            Camera::default(),
            Msaa::Sample4,
            Projection::from(PerspectiveProjection::default()),
        ))
        .id();
    app.update();
    commit(&mut app, view);
    assert_eq!(
        app.world().entity(view).get::<Msaa>(),
        Some(&Msaa::Sample4),
        "the camera is actually multisampled before the request"
    );
    app.world_mut()
        .resource_mut::<AaPolicy>()
        .request(AaMode::Off);
    app.update();
    assert_eq!(
        app.world().resource::<AaPolicy>().effective(),
        AaEffective::Msaa { samples: 4 },
        "Off cannot be effective while the camera still has four samples"
    );
    assert_eq!(
        app.world().resource::<AaPolicy>().reason(),
        AaReason::OffUnavailable
    );
    assert_eq!(
        app.world().entity(view).get::<Msaa>(),
        Some(&Msaa::Sample4),
        "this foundation does not change the camera's MSAA"
    );
    assert!(
        app.world().resource::<TemporalHistory>().is_valid(view),
        "the Off request fell back to the same effective MSAA path"
    );
}

/// Each explicit notification is a real transition, even without an intervening commit.
#[test]
fn repeated_notifications_advance_uncommitted_history() {
    for reason in [
        AaInvalidation::CameraCut,
        AaInvalidation::Teleport,
        AaInvalidation::LoadingCover,
        AaInvalidation::WorldTransition,
    ] {
        let mut app = App::new();
        app.init_resource::<TemporalHistory>();
        let targeted = view(&mut app);
        let other = view(&mut app);
        let (start, other_start) = {
            let mut history = app.world_mut().resource_mut::<TemporalHistory>();
            history.register(targeted, TemporalTarget::default());
            history.register(other, TemporalTarget::default());
            (
                history.view(targeted).unwrap().generation(),
                history.view(other).unwrap().generation(),
            )
        };

        for step in 1..=2 {
            app.world_mut()
                .resource_mut::<TemporalHistory>()
                .invalidate(targeted, reason);
            let history = app.world().resource::<TemporalHistory>();
            assert_eq!(
                history.view(targeted).unwrap().generation(),
                start + step,
                "{reason:?} notification {step} must advance the generation"
            );
            assert!(!history.is_valid(targeted));
            assert_eq!(history.view(targeted).unwrap().last_invalidation(), reason);
            assert_eq!(
                history.view(other).unwrap().generation(),
                other_start,
                "{reason:?} must not touch an un-targeted view"
            );
        }
    }
}

/// Global notifications are also transitions, not duplicate-state suppression.
#[test]
fn repeated_global_notifications_advance_each_registered_view() {
    for reason in [AaInvalidation::CameraCut, AaInvalidation::Teleport] {
        let mut app = App::new();
        app.init_resource::<TemporalHistory>();
        let a = view(&mut app);
        let b = view(&mut app);
        let (start_a, start_b) = {
            let mut history = app.world_mut().resource_mut::<TemporalHistory>();
            history.register(a, TemporalTarget::default());
            history.register(b, TemporalTarget::default());
            (
                history.view(a).unwrap().generation(),
                history.view(b).unwrap().generation(),
            )
        };

        for step in 1..=2 {
            app.world_mut()
                .resource_mut::<TemporalHistory>()
                .invalidate_all(reason);
            let history = app.world().resource::<TemporalHistory>();
            assert_eq!(
                history.view(a).unwrap().generation(),
                start_a + step,
                "{reason:?} global notification {step} must advance view a"
            );
            assert_eq!(
                history.view(b).unwrap().generation(),
                start_b + step,
                "{reason:?} global notification {step} must advance view b"
            );
            assert!(!history.is_valid(a));
            assert!(!history.is_valid(b));
            assert_eq!(history.view(a).unwrap().last_invalidation(), reason);
            assert_eq!(history.view(b).unwrap().last_invalidation(), reason);
        }
    }
}

/// Both loading-cover edges invalidate; an unchanged cover state does not.
#[test]
fn loading_cover_edges_invalidate_without_intervening_commits() {
    let mut app = test_app();
    let view = app
        .world_mut()
        .spawn((
            WorldCamera,
            Camera::default(),
            Msaa::Off,
            Projection::from(PerspectiveProjection::default()),
            TemporalTarget::default(),
        ))
        .id();
    app.update();
    let start = app
        .world()
        .resource::<TemporalHistory>()
        .view(view)
        .unwrap()
        .generation();

    app.world_mut().resource_mut::<Viewer>().world_covered = true;
    app.update();
    let covered = app
        .world()
        .resource::<TemporalHistory>()
        .view(view)
        .unwrap()
        .generation();
    assert_eq!(covered, start + 1, "the cover-up edge must invalidate");
    assert!(!app.world().resource::<TemporalHistory>().is_valid(view));

    app.world_mut().resource_mut::<Viewer>().world_covered = false;
    app.update();
    let revealed = app
        .world()
        .resource::<TemporalHistory>()
        .view(view)
        .unwrap()
        .generation();
    assert_eq!(
        revealed,
        covered + 1,
        "the reveal edge must invalidate without a commit"
    );
    assert!(!app.world().resource::<TemporalHistory>().is_valid(view));

    app.update();
    assert_eq!(
        app.world()
            .resource::<TemporalHistory>()
            .view(view)
            .unwrap()
            .generation(),
        revealed,
        "an unchanged cover state must preserve the generation"
    );
}

fn test_app() -> App {
    let mut app = App::new();
    // The literal, not `MsaaSetting::default()`, which reads `$WOW_MSAA`.
    app.insert_resource(MsaaSetting { samples: 1 })
        .init_resource::<Viewer>()
        .init_resource::<WorldLive>()
        .add_message::<MapChange>()
        .add_plugins(AaPlugin);
    app
}
