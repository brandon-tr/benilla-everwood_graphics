use super::*;
use benilla_world::aa::{AaPlugin, TemporalHistory};
use benilla_world::schedule::WorldLive;
use benilla_world::view::{MsaaSetting, Viewer};
use benilla_world::world_map::MapChange;
use bevy::render::view::Msaa;

#[test]
fn the_spyglass_narrows_our_own_field_of_view_six_fold() {
    let scoped = ScopedView {
        zoom: Some(15.0 / REFERENCE_DEFAULT_DEGREES),
    };
    let fov = CAM_FOVY * scoped.zoom.unwrap();
    assert!(
        (fov - CAM_FOVY / 6.0).abs() < 1e-6,
        "the spyglass is a 6x zoom off whatever our normal FOV is, not a literal 15 degrees"
    );
    assert!(
        (fov.to_degrees() - 7.5).abs() < 1e-4,
        "…which lands at 7.5 degrees vertical for our 45 degree default, not 15"
    );
    assert!(scoped.active());

    let none = ScopedView { zoom: None };
    assert_eq!(
        CAM_FOVY * none.zoom.unwrap_or(1.0),
        CAM_FOVY,
        "unscoped restores OUR default, never the reference's stored 90"
    );
    assert!(!none.active());
}

/// The production writer mutably pattern-matches the projection every in-world frame and so
/// marks it changed even with no scope. Equal values must not invalidate committed history.
#[test]
fn unscoped_camera_updates_preserve_temporal_history() {
    let mut app = App::new();
    app.insert_resource(MsaaSetting { samples: 1 })
        .init_resource::<Viewer>()
        .init_resource::<WorldLive>()
        .init_resource::<ScopedView>()
        .add_message::<MapChange>()
        .add_plugins(AaPlugin)
        .add_systems(Update, apply_scoped_view);
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
    let generation = {
        let mut history = app.world_mut().resource_mut::<TemporalHistory>();
        let token = history.begin(view).expect("registered");
        assert!(history.commit(token));
        history.view(view).unwrap().generation()
    };

    for frame in 0..3 {
        let before = app.world().read_change_tick();
        app.update();
        let now = app.world().read_change_tick();
        assert!(
            app.world()
                .entity(view)
                .get_ref::<Projection>()
                .unwrap()
                .last_changed()
                .is_newer_than(before, now),
            "frame {frame}: apply_scoped_view must mark the projection changed"
        );
        let history = app.world().resource::<TemporalHistory>();
        assert!(
            history.is_valid(view),
            "frame {frame}: the equal scoped-view write must preserve history"
        );
        assert_eq!(
            history.view(view).unwrap().generation(),
            generation,
            "frame {frame}: the equal scoped-view write must not bump the generation"
        );
    }
}
