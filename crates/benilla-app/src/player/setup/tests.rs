use super::*;
use crate::net::TeleportMessage;
use benilla_world::aa::TemporalTarget;

/// The app-owned same-map teleport cuts the view. A frame without one keeps the pair.
#[test]
fn a_same_map_teleport_invalidates_temporal_history() {
    let mut app = App::new();
    app.add_message::<TeleportMessage>()
        .add_message::<crate::net::SelfMoveMessage>()
        .init_resource::<TemporalHistory>();
    let view = app.world_mut().spawn_empty().id();
    {
        let mut history = app.world_mut().resource_mut::<TemporalHistory>();
        history.register(view, TemporalTarget::default());
        let token = history.begin(view).expect("registered");
        assert!(history.commit(token));
    }
    app.add_systems(Last, invalidate_history_on_teleport);
    app.update();
    assert!(
        app.world().resource::<TemporalHistory>().is_valid(view),
        "a frame with no teleport leaves the pair alone"
    );

    app.world_mut().write_message(TeleportMessage {
        guid: 1,
        counter: 0,
        position: [0.0; 3],
        orientation: 0.0,
    });
    app.update();
    assert!(!app.world().resource::<TemporalHistory>().is_valid(view));
    assert_eq!(
        app.world()
            .resource::<TemporalHistory>()
            .view(view)
            .unwrap()
            .last_invalidation(),
        AaInvalidation::Teleport
    );
}

use crate::aa_options::AaOptionsPlugin;
use benilla_world::aa::{
    AaEffective, AaInvalidation, AaMode, AaPlugin, AaPolicy, AaReadiness, TemporalHistory,
    TemporalResolveView,
};
use benilla_world::schedule::WorldLive;
use benilla_world::view::{MsaaSetting, Viewer};
use benilla_world::world_map::MapChange;
use bevy::render::camera::TemporalJitter;

/// Boot the player with one `gxAntiAliasing` value, exactly as `GamePlugins` orders it: the option
/// snapshot and its mode application before `setup_player`.
fn boot_app(value: &str, samples: u32) -> App {
    let mut app = App::new();
    let mut cvars = crate::cvars::Cvars::default();
    cvars.set("gxAntiAliasing", value);
    app.insert_resource(cvars)
        .insert_resource(MsaaSetting { samples })
        .init_resource::<Viewer>()
        .init_resource::<WorldLive>()
        .add_message::<MapChange>()
        .add_plugins(AaPlugin)
        .add_plugins(AaOptionsPlugin)
        .add_systems(
            Startup,
            setup_player.after(crate::aa_options::AaBootApplied),
        );
    app
}

fn world_camera(app: &mut App) -> Entity {
    let mut query = app
        .world_mut()
        .query_filtered::<Entity, With<WorldCamera>>();
    query.single(app.world()).expect("one world camera")
}

/// `gxAntiAliasing = 2`: the real world camera runs the native path — readiness and policy both
/// request TAA, the camera is single-sample, the resolve marker is present and jitter is scheduled.
#[test]
fn gx_anti_aliasing_two_selects_the_native_path() {
    let mut app = boot_app("2", 4);
    app.update();
    assert!(app.world().resource::<AaReadiness>().native_taa());
    let policy = *app.world().resource::<AaPolicy>();
    assert_eq!(policy.requested(), AaMode::Taa);
    assert_eq!(policy.effective(), AaEffective::Taa);
    assert_eq!(policy.reason(), benilla_world::aa::AaReason::TaaRequested);

    let view = world_camera(&mut app);
    assert_eq!(*app.world().get::<Msaa>(view).unwrap(), Msaa::Off);
    assert!(app.world().get::<TemporalResolveView>(view).is_some());
    assert!(app.world().get::<TemporalJitter>(view).is_some());

    // The first offset is the sequence's zero; the next frame proves a live schedule.
    app.update();
    assert_ne!(
        app.world().get::<TemporalJitter>(view).unwrap().offset,
        Vec2::ZERO
    );
}

/// A camera cut (or any invalidation) restarts the jitter sequence: the first sample is the zero
/// offset again, so the reconstructed frame is centred before the pattern resumes.
#[test]
fn a_camera_cut_restarts_the_jitter_sequence() {
    let mut app = boot_app("2", 4);
    app.update();
    app.update();
    let view = world_camera(&mut app);
    assert_ne!(
        app.world().get::<TemporalJitter>(view).unwrap().offset,
        Vec2::ZERO
    );
    app.world_mut()
        .resource_mut::<TemporalHistory>()
        .invalidate(view, AaInvalidation::CameraCut);
    app.update();
    assert_eq!(
        app.world().get::<TemporalJitter>(view).unwrap().offset,
        Vec2::ZERO,
        "a cut restarts the sequence"
    );
}

/// `gxAntiAliasing = 1` keeps the established MSAA path byte for byte: policy, camera samples, no
/// marker and no jitter.
#[test]
fn gx_anti_aliasing_one_keeps_msaa() {
    let mut app = boot_app("1", 4);
    app.update();
    assert!(!app.world().resource::<AaReadiness>().native_taa());
    let policy = *app.world().resource::<AaPolicy>();
    assert_eq!(policy.requested(), AaMode::Msaa);
    assert_eq!(policy.effective(), AaEffective::Msaa { samples: 4 });

    let view = world_camera(&mut app);
    assert_eq!(*app.world().get::<Msaa>(view).unwrap(), Msaa::Sample4);
    assert!(app.world().get::<TemporalResolveView>(view).is_none());
    assert!(app.world().get::<TemporalJitter>(view).is_none());
}

/// `gxAntiAliasing = 0` keeps the existing Off behavior: single-sample, no marker, no jitter.
#[test]
fn gx_anti_aliasing_zero_keeps_off() {
    let mut app = boot_app("0", 4);
    app.update();
    assert!(!app.world().resource::<AaReadiness>().native_taa());
    let policy = *app.world().resource::<AaPolicy>();
    assert_eq!(policy.requested(), AaMode::Msaa);
    assert_eq!(policy.effective(), AaEffective::Msaa { samples: 1 });

    let view = world_camera(&mut app);
    assert_eq!(*app.world().get::<Msaa>(view).unwrap(), Msaa::Off);
    assert!(app.world().get::<TemporalResolveView>(view).is_none());
    assert!(app.world().get::<TemporalJitter>(view).is_none());
}
