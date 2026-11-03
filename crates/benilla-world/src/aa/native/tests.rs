use super::*;
use crate::aa::{
    AaInvalidation, AaMode, AaPlugin, AaReadiness, AaReason, TemporalResolveView, TemporalTarget,
};
use crate::schedule::WorldLive;
use crate::view::{MsaaSetting, Viewer};
use crate::world_map::MapChange;
use bevy::render::view::Msaa;

fn test_app(selection: AaMode, native_ready: bool) -> App {
    let mut app = App::new();
    app.insert_resource(MsaaSetting { samples: 1 })
        .init_resource::<Viewer>()
        .init_resource::<WorldLive>()
        .add_message::<MapChange>()
        .add_plugins(AaPlugin);
    app.world_mut()
        .resource_mut::<AaReadiness>()
        .set_native_taa(native_ready);
    app.world_mut()
        .resource_mut::<AaPolicy>()
        .request(selection);
    app
}

fn spawn_world_camera(app: &mut App, msaa: Msaa, marked: bool) -> Entity {
    let mut entity = app.world_mut().spawn((
        WorldCamera,
        Camera::default(),
        msaa,
        TemporalTarget::new(UVec2::new(64, 64), 1.0),
    ));
    if marked {
        entity.insert(TemporalResolveView);
    }
    entity.id()
}

fn generation(app: &App, view: Entity) -> u64 {
    app.world()
        .resource::<TemporalHistory>()
        .view(view)
        .expect("registered view")
        .generation()
}

fn bridge(app: &App) -> &NativeResolveBridge {
    app.world().resource::<NativeResolveBridge>()
}

/// A successful encode for a frame is committed once a later extraction proves the render frame
/// finished; before that it stays in flight.
#[test]
fn a_successful_encode_commits_after_a_later_extraction() {
    let mut app = test_app(AaMode::Taa, true);
    let view = spawn_world_camera(&mut app, Msaa::Off, true);
    app.update();
    let token = app
        .world()
        .resource::<TemporalHistory>()
        .in_flight(view)
        .expect("the marker camera begins a token");
    let generation = generation(&app, view);
    let frame = token.frame();

    let bridge = bridge(&app).clone();
    bridge.note_extraction(1);
    bridge.attempt(view, 1, generation);
    bridge.mark_committed(view, 1, generation);
    assert_eq!(
        bridge.settle(view, frame),
        Settle::Pending,
        "the owning render frame may still be running at extraction 1"
    );

    bridge.note_extraction(2);
    app.update();
    let history = app.world().resource::<TemporalHistory>();
    assert!(history.is_valid(view));
    assert_eq!(history.view(view).unwrap().last_commit_frame(), Some(frame));
    assert!(
        history.in_flight(view).is_some(),
        "the next frame's token is begun after the settled one"
    );
}

/// A frame whose resolve did not encode must abandon, never commit the previous pair.
#[test]
fn a_missing_or_failed_resolve_abandons() {
    for commit in [false, true] {
        let mut app = test_app(AaMode::Taa, true);
        let view = spawn_world_camera(&mut app, Msaa::Off, true);
        app.update();
        let generation = generation(&app, view);
        let bridge = bridge(&app).clone();
        bridge.note_extraction(1);
        bridge.attempt(view, 1, generation);
        if commit {
            bridge.mark_committed(view, 1, generation);
        }
        bridge.note_extraction(2);
        app.update();
        let history = app.world().resource::<TemporalHistory>();
        if commit {
            assert!(history.is_valid(view), "an encoded resolve commits");
        } else {
            assert!(!history.is_valid(view), "a failed frame never commits");
            assert_eq!(
                history.view(view).unwrap().last_invalidation(),
                AaInvalidation::FrameAbandoned
            );
        }
    }
}

/// Extraction moving past a begun frame without a request (a gated/inactive view, a multisampled
/// camera or a missing target) abandons it.
#[test]
fn a_frame_the_renderer_never_attempted_abandons() {
    let mut app = test_app(AaMode::Taa, true);
    let view = spawn_world_camera(&mut app, Msaa::Off, true);
    app.update();
    let bridge = bridge(&app).clone();
    bridge.note_extraction(1);
    app.update();
    let history = app.world().resource::<TemporalHistory>();
    assert!(!history.is_valid(view));
    assert_eq!(
        history.view(view).unwrap().last_invalidation(),
        AaInvalidation::FrameAbandoned
    );
}

/// A commit stamped with a stale generation publishes nothing, and the frame abandons.
#[test]
fn a_stale_generation_cannot_commit() {
    let mut app = test_app(AaMode::Taa, true);
    let view = spawn_world_camera(&mut app, Msaa::Off, true);
    app.update();
    let generation = generation(&app, view);
    let bridge = bridge(&app).clone();
    bridge.note_extraction(1);
    bridge.attempt(view, 1, generation);
    bridge.mark_committed(view, 1, generation + 1);
    bridge.note_extraction(2);
    app.update();
    let history = app.world().resource::<TemporalHistory>();
    assert!(!history.is_valid(view));
    assert_eq!(
        history.view(view).unwrap().last_invalidation(),
        AaInvalidation::FrameAbandoned
    );
}

/// An invalidation drops the in-flight token, so a later render success cannot resurrect history.
#[test]
fn invalidation_cannot_be_resurrected_by_a_late_commit() {
    let mut app = test_app(AaMode::Taa, true);
    let view = spawn_world_camera(&mut app, Msaa::Off, true);
    app.update();
    let token = app
        .world()
        .resource::<TemporalHistory>()
        .in_flight(view)
        .expect("begun");
    let generation = generation(&app, view);
    app.world_mut()
        .resource_mut::<TemporalHistory>()
        .invalidate(view, AaInvalidation::CameraCut);
    let bridge = bridge(&app).clone();
    bridge.note_extraction(1);
    bridge.attempt(view, 1, generation);
    bridge.mark_committed(view, 1, generation);
    bridge.note_extraction(2);
    app.update();
    let history = app.world().resource::<TemporalHistory>();
    assert!(!history.is_valid(view));
    assert!(
        !app.world_mut()
            .resource_mut::<TemporalHistory>()
            .commit(token),
        "the cut token is not current"
    );
}

/// Readiness off: a TAA request resolves to MSAA, and no marker or history token appears.
#[test]
fn readiness_off_falls_back_without_marker_or_tokens() {
    let mut app = test_app(AaMode::Taa, false);
    let view = spawn_world_camera(&mut app, Msaa::Sample4, true);
    app.update();
    let policy = *app.world().resource::<AaPolicy>();
    assert_eq!(policy.effective(), AaEffective::Msaa { samples: 4 });
    assert_eq!(policy.reason(), AaReason::TaaUnavailable);
    assert!(app.world().get::<TemporalResolveView>(view).is_none());
    let history = app.world().resource::<TemporalHistory>();
    assert!(history.is_registered(view));
    assert!(history.in_flight(view).is_none());
    assert!(!history.is_valid(view));
}

/// The marker follows the effective mode: TAA marks, MSAA unmarks.
#[test]
fn the_marker_follows_the_effective_mode() {
    let mut app = test_app(AaMode::Taa, true);
    let view = spawn_world_camera(&mut app, Msaa::Off, false);
    app.update();
    assert!(app.world().get::<TemporalResolveView>(view).is_some());

    app.world_mut()
        .resource_mut::<AaPolicy>()
        .request(AaMode::Msaa);
    app.update();
    assert!(app.world().get::<TemporalResolveView>(view).is_none());
    assert_eq!(
        app.world().resource::<AaPolicy>().effective(),
        AaEffective::Msaa { samples: 1 }
    );
}

/// A replaced world camera is marked and registered; the removed entity keeps neither.
#[test]
fn a_replaced_camera_is_marked_afresh() {
    let mut app = test_app(AaMode::Taa, true);
    let first = spawn_world_camera(&mut app, Msaa::Off, false);
    app.update();
    assert!(app.world().get::<TemporalResolveView>(first).is_some());

    app.world_mut().entity_mut(first).despawn();
    let second = spawn_world_camera(&mut app, Msaa::Off, false);
    app.update();
    assert!(app.world().get::<TemporalResolveView>(second).is_some());
    let history = app.world().resource::<TemporalHistory>();
    assert!(history.is_registered(second));
    assert!(!history.is_registered(first));
}

/// The player-UI camera is a `Camera2d` without `WorldCamera`: never marked, never registered in
/// world history.
#[test]
fn a_ui_camera_is_never_marked_or_registered() {
    let mut app = test_app(AaMode::Taa, true);
    let _world = spawn_world_camera(&mut app, Msaa::Off, false);
    let ui = app.world_mut().spawn((Camera2d, Msaa::Off)).id();
    app.update();
    assert!(app.world().get::<TemporalResolveView>(ui).is_none());
    assert!(!app.world().resource::<TemporalHistory>().is_registered(ui));
}

/// The bridge is only a channel: a commit for an epoch that was never attempted does nothing.
#[test]
fn a_commit_without_an_attempt_is_ignored() {
    let bridge = NativeResolveBridge::default();
    let view = Entity::from_raw_u32(7).unwrap();
    bridge.begin_frame(view, 1);
    bridge.mark_committed(view, 3, 1);
    bridge.note_extraction(4);
    assert_eq!(bridge.settle(view, 1), Settle::Abandoned);
}
