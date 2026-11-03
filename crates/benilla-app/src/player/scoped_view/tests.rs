use super::*;
use benilla_world::aa::{AaPlugin, TemporalHistory};
use benilla_world::schedule::WorldLive;
use benilla_world::view::WorldCamera;
use benilla_world::view::{MsaaSetting, Viewer};
use benilla_world::world_map::MapChange;
use bevy::render::view::Msaa;

#[test]
fn the_spyglass_narrows_our_own_field_of_view_six_fold() {
    use benilla_world::view::world_fovy;
    let scoped = ScopedView {
        zoom: Some(15.0 / REFERENCE_DEFAULT_DEGREES),
        ..Default::default()
    };
    let normal = world_fovy(16.0 / 9.0);
    let fov = normal * scoped.zoom.unwrap();
    assert!(
        (fov - normal / 6.0).abs() < 1e-6,
        "the spyglass is a 6x zoom off whatever the normal FOV is, not a literal 15 degrees"
    );
    assert!(
        (fov.to_degrees() - 44.1 / 6.0).abs() < 0.02,
        "…which lands at a sixth of the 16:9 field of view, not 15"
    );
    assert!(scoped.active());

    let none = ScopedView::default();
    assert_eq!(
        none.zoom.unwrap_or(1.0),
        1.0,
        "unscoped restores the reference's 90, never a stored default"
    );
    assert!(!none.active());
}

/// The production writers run every in-world frame: `apply_scoped_view` sets `FovZoom` and
/// `stamp_world_fov` compares before it touches the projection, so an unscoped frame rewrites
/// nothing. An equal write must not invalidate the committed temporal history either (the earlier
/// mutable pattern-match marked the projection changed every frame; TAA had to survive that).
#[test]
fn unscoped_camera_updates_preserve_temporal_history() {
    let mut app = App::new();
    app.insert_resource(MsaaSetting { samples: 1 })
        .init_resource::<Viewer>()
        .init_resource::<WorldLive>()
        .init_resource::<ScopedView>()
        // The production writer takes `FovZoom` too (the reference's held `[cam+0x40]` fraction).
        .init_resource::<FovZoom>()
        .add_message::<MapChange>()
        .add_plugins(AaPlugin)
        .add_systems(
            Update,
            (apply_scoped_view, benilla_world::view::stamp_world_fov),
        );
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
            !app.world()
                .entity(view)
                .get_ref::<Projection>()
                .unwrap()
                .last_changed()
                .is_newer_than(before, now),
            "frame {frame}: an equal scoped-view frame rewrote the projection"
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

const SPYGLASS: u32 = 12883;
const HELD: Option<(u32, f32)> = Some((SPYGLASS, 15.0));

/// The aura's add starts the scope and its removal ends it (`0x5ff462`, `0x61236e`).
#[test]
fn the_scope_moves_on_a_change_of_the_aura() {
    let mut scoped = ScopedView::default();
    scoped.observe(HELD, false);
    assert_eq!(scoped.zoom, Some(15.0 / REFERENCE_DEFAULT_DEGREES));
    scoped.observe(HELD, false);
    assert!(scoped.active());
    scoped.observe(None, false);
    assert!(!scoped.active());
}

/// The create's `PostInit` starts the scope (`0x5fbb3c`) and the arrival ends it (`0x5deb44`),
/// and here the standing aura starts nothing until it changes (`0x604d00`).
#[test]
fn an_arrival_ends_the_scope_under_a_standing_aura() {
    let mut scoped = ScopedView::default();
    scoped.observe(HELD, false);
    scoped.observe(HELD, true);
    assert!(!scoped.active());
    assert!(scoped.arrival_end);
    scoped.observe(HELD, false);
    assert!(!scoped.active());
    assert!(!scoped.arrival_end);
    scoped.observe(None, false);
    scoped.observe(HELD, false);
    assert!(scoped.active(), "a new add starts it");
}

/// With no aura on the create nothing started, so no view is owed at this arrival.
#[test]
fn an_arrival_with_no_aura_owes_no_view() {
    let mut scoped = ScopedView::default();
    scoped.observe(None, true);
    assert!(!scoped.arrival_end);
}

/// A new camera holds no scope, building it shows no aura, and it selects no view.
#[test]
fn a_new_camera_ends_the_scope_under_a_standing_aura() {
    let mut scoped = ScopedView::default();
    scoped.observe(HELD, false);
    scoped.new_camera();
    assert!(!scoped.active());
    scoped.observe(HELD, false);
    assert!(!scoped.active());
    assert!(!scoped.arrival_end);
}

fn store(fields: &[(u16, u32)]) -> ObjectStore {
    ObjectStore(benilla_protocol::ObjectFields::from_pairs(fields))
}

/// Our create under the aura holds no scope; the aura's later add does, and zooms.
#[test]
fn a_create_under_the_aura_holds_no_scope_and_a_later_add_does() {
    use benilla_formats::{SpellCatalog, SpellDisplay};
    // `UNIT_FIELD_AURA` slot 0 and its `AURAFLAGS` nibble.
    let aura = [(47, SPYGLASS), (95, 8)];
    let mut world = World::new();
    world.init_resource::<ScopedView>();
    world.init_resource::<FovZoom>();
    world.insert_resource(Spells {
        catalog: SpellCatalog::from_displays(std::collections::HashMap::from([(
            SPYGLASS,
            SpellDisplay {
                effect_apply_aura: [SPELL_AURA_FAR_SIGHT, 0, 0],
                effect_misc_value: [15, 0, 0],
                ..Default::default()
            },
        )])),
        ..Spells::empty_for_tests()
    });
    let apply = world.register_system(apply_scoped_view);
    let me = world.spawn((store(&aura), SelfPlayer)).id();
    world.run_system(apply).expect("the look runs");
    assert!(world.resource::<ScopedView>().arrival_end);
    assert!(!world.resource::<ScopedView>().active());
    world.run_system(apply).expect("the look runs");
    assert!(!world.resource::<ScopedView>().arrival_end);
    assert!(!world.resource::<ScopedView>().active());
    world.entity_mut(me).insert(store(&[]));
    world.run_system(apply).expect("the look runs");
    world.entity_mut(me).insert(store(&aura));
    world.run_system(apply).expect("the look runs");
    assert!(world.resource::<ScopedView>().active());
    let zoom = world.resource::<FovZoom>().0;
    assert_eq!(zoom, 15.0 / REFERENCE_DEFAULT_DEGREES);
}
