//! The world-view demand scan: what qualifies, what is excluded, and that first warming needs no
//! prepared attachment.

use super::*;
use crate::aa::{AaMode, AaPolicy, AaReadiness, TemporalTarget};
use bevy::camera::{RenderTargetInfo, SubCameraView, Viewport};

fn demand_app() -> App {
    let mut app = App::new();
    app.init_resource::<RigTaaWorldDemand>()
        .add_systems(Last, extract_rig_taa_world_views);
    app
}

/// A camera whose render target the main-world camera system has resolved.
fn resolved_camera(size: UVec2) -> Camera {
    let mut camera = Camera::default();
    camera.computed.target_info = Some(RenderTargetInfo {
        physical_size: size,
        scale_factor: 1.0,
    });
    camera
}

/// A fully compatible 3D world view: a resolved 640x480 target, no attachments, the mount's
/// default `reset`.
fn eligible_view(app: &mut App) -> Entity {
    app.world_mut()
        .spawn((
            Camera3d::default(),
            resolved_camera(UVec2::new(640, 480)),
            Msaa::Off,
            WorldCamera,
            TemporalAntiAliasing::default(),
        ))
        .id()
}

fn views(app: &mut App) -> Vec<Entity> {
    app.world()
        .resource::<RigTaaWorldDemand>()
        .main_views
        .clone()
}

/// A minimal world view with Bevy's native-TAA mount is demand, and nothing else is needed: the
/// component only exists here, no prepass or history attachment has ever been created, and the
/// mount's own `reset` still holds its default `true` — the state a first frame is in.
#[test]
fn a_native_taa_world_view_is_demand_before_any_attachment_exists() {
    let mut app = demand_app();
    let view = eligible_view(&mut app);
    let mount = app
        .world()
        .get::<TemporalAntiAliasing>(view)
        .expect("the mount is on the camera");
    assert!(mount.reset, "first warming must not wait for reset = false");
    assert!(
        app.get_sub_app(bevy::render::RenderApp).is_none(),
        "a main-world-only app has no render world, so no prepass or history attachment exists"
    );

    app.update();
    assert_eq!(views(&mut app), vec![view]);

    // A later frame with no change and an unchanged set publishes nothing new.
    app.update();
    assert_eq!(views(&mut app), vec![view]);
    assert!(!app.world().resource::<RigTaaWorldDemand>().is_empty());
}

/// Two compatible views share one demand set; the snapshot itself is world-scope.
///
/// The copy and record stages receive demand as one boolean (`!is_empty()`), so a second eligible
/// view cannot start a second rotation for the same epoch — there is one palette, one rig table
/// and one frame block to snapshot, not one per view.
#[test]
fn every_eligible_view_shares_the_same_demand() {
    let mut app = demand_app();
    let first = eligible_view(&mut app);
    let second = eligible_view(&mut app);
    app.update();
    let mut want = vec![first, second];
    want.sort_unstable_by_key(|view| view.to_bits());
    assert_eq!(views(&mut app), want);
}

/// A stored `Taa` preference is not demand: only the view's mount is.
#[test]
fn a_saved_preference_without_the_mount_is_not_demand() {
    let mut app = demand_app();
    app.insert_resource(AaReadiness {
        taa: true,
        temporal_upscale: false,
    })
    .insert_resource(AaPolicy::default());
    app.world_mut()
        .resource_mut::<AaPolicy>()
        .request(AaMode::Taa);
    let _ = app
        .world_mut()
        .spawn((
            Camera3d::default(),
            resolved_camera(UVec2::new(640, 480)),
            Msaa::Off,
            WorldCamera,
            DepthPrepass,
            MotionVectorPrepass,
        ))
        .id();
    app.update();
    assert!(views(&mut app).is_empty());
}

/// `WorldCamera` is the world-view marker; portrait booths are `Camera3d`s without it.
#[test]
fn a_portrait_camera_is_not_a_world_view() {
    let mut app = demand_app();
    let _ = app
        .world_mut()
        .spawn((
            Camera3d::default(),
            resolved_camera(UVec2::new(256, 256)),
            Msaa::Off,
            TemporalAntiAliasing::default(),
        ))
        .id();
    app.update();
    assert!(views(&mut app).is_empty());
}

/// The water mirror is a world camera by accident of construction in no code path; if one ever
/// carried the marker, the mirror's own component still keeps it out.
#[test]
fn a_mirror_camera_is_not_a_world_view() {
    let mut app = demand_app();
    let view = eligible_view(&mut app);
    app.world_mut().entity_mut(view).insert(StaticGxMirror);
    app.update();
    assert!(views(&mut app).is_empty());
}

/// Each incompatibility keeps the view out on its own.
#[test]
fn incompatible_views_are_excluded() {
    // MSAA on.
    let mut app = demand_app();
    let view = eligible_view(&mut app);
    app.world_mut().entity_mut(view).insert(Msaa::Sample4);
    app.update();
    assert!(views(&mut app).is_empty(), "MSAA on");

    // Inactive.
    let mut app = demand_app();
    let view = eligible_view(&mut app);
    app.world_mut()
        .entity_mut(view)
        .get_mut::<Camera>()
        .unwrap()
        .is_active = false;
    app.update();
    assert!(views(&mut app).is_empty(), "inactive camera");

    // Deferred prepass.
    let mut app = demand_app();
    let view = eligible_view(&mut app);
    app.world_mut().entity_mut(view).insert(DeferredPrepass);
    app.update();
    assert!(views(&mut app).is_empty(), "deferred prepass");

    // Resolution override.
    let mut app = demand_app();
    let view = eligible_view(&mut app);
    app.world_mut()
        .entity_mut(view)
        .insert(MainPassResolutionOverride(UVec2::new(320, 240)));
    app.update();
    assert!(views(&mut app).is_empty(), "resolution override");

    // Partial viewport.
    let mut app = demand_app();
    let view = eligible_view(&mut app);
    app.world_mut()
        .entity_mut(view)
        .get_mut::<Camera>()
        .unwrap()
        .viewport = Some(Viewport {
        physical_position: UVec2::ONE,
        physical_size: UVec2::new(320, 240),
        ..default()
    });
    app.update();
    assert!(views(&mut app).is_empty(), "partial viewport");

    // Sub-camera view.
    let mut app = demand_app();
    let view = eligible_view(&mut app);
    app.world_mut()
        .entity_mut(view)
        .get_mut::<Camera>()
        .unwrap()
        .sub_camera_view = Some(SubCameraView {
        full_size: UVec2::new(640, 480),
        offset: Vec2::ZERO,
        size: UVec2::new(320, 240),
    });
    app.update();
    assert!(views(&mut app).is_empty(), "sub-camera view");

    // An unresolved target: the camera system never filled `computed.target_info`.
    let mut app = demand_app();
    let view = eligible_view(&mut app);
    app.world_mut()
        .entity_mut(view)
        .get_mut::<Camera>()
        .unwrap()
        .computed
        .target_info = None;
    app.update();
    assert!(views(&mut app).is_empty(), "unresolved target");

    // A resolved target with a zero physical dimension.
    let mut app = demand_app();
    let view = eligible_view(&mut app);
    app.world_mut()
        .entity_mut(view)
        .get_mut::<Camera>()
        .unwrap()
        .computed
        .target_info = Some(RenderTargetInfo {
        physical_size: UVec2::new(640, 0),
        scale_factor: 1.0,
    });
    app.update();
    assert!(views(&mut app).is_empty(), "zero height");

    // Not a 3D camera.
    let mut app = demand_app();
    let view = eligible_view(&mut app);
    app.world_mut().entity_mut(view).remove::<Camera3d>();
    app.update();
    assert!(views(&mut app).is_empty(), "not a 3D camera");

    // No depth or motion prepass (the mount's own requirements, validated rather than assumed).
    let mut app = demand_app();
    let view = eligible_view(&mut app);
    app.world_mut().entity_mut(view).remove::<DepthPrepass>();
    app.update();
    assert!(views(&mut app).is_empty(), "no depth prepass");

    let mut app = demand_app();
    let view = eligible_view(&mut app);
    app.world_mut()
        .entity_mut(view)
        .remove::<MotionVectorPrepass>();
    app.update();
    assert!(views(&mut app).is_empty(), "no motion prepass");
}

/// The predicate itself, including the degenerate target shapes a clamped `TemporalTarget` can
/// never express.
#[test]
fn the_eligibility_predicate_rejects_each_incompatibility() {
    let eligible = RigTaaViewFacts {
        native_taa: true,
        camera_active: true,
        three_d: true,
        msaa_off: true,
        depth_prepass: true,
        motion_prepass: true,
        resolved_target: Some(UVec2::new(64, 64)),
        ..default()
    };
    assert!(rig_taa_view_is_eligible(eligible));
    let rejected = [
        RigTaaViewFacts {
            native_taa: false,
            ..eligible
        },
        RigTaaViewFacts {
            camera_active: false,
            ..eligible
        },
        RigTaaViewFacts {
            three_d: false,
            ..eligible
        },
        RigTaaViewFacts {
            msaa_off: false,
            ..eligible
        },
        RigTaaViewFacts {
            depth_prepass: false,
            ..eligible
        },
        RigTaaViewFacts {
            motion_prepass: false,
            ..eligible
        },
        RigTaaViewFacts {
            deferred_prepass: true,
            ..eligible
        },
        RigTaaViewFacts {
            resolution_override: true,
            ..eligible
        },
        RigTaaViewFacts {
            viewport: true,
            ..eligible
        },
        RigTaaViewFacts {
            sub_view: true,
            ..eligible
        },
        RigTaaViewFacts {
            resolved_target: None,
            ..eligible
        },
        RigTaaViewFacts {
            resolved_target: Some(UVec2::ZERO),
            ..eligible
        },
        RigTaaViewFacts {
            resolved_target: Some(UVec2::new(0, 64)),
            ..eligible
        },
        RigTaaViewFacts {
            resolved_target: Some(UVec2::new(64, 0)),
            ..eligible
        },
    ];
    for facts in rejected {
        assert!(!rig_taa_view_is_eligible(facts), "{facts:?}");
    }
}

/// Removing the mount ends demand; adding it back restores it in the same run.
#[test]
fn demand_tracks_the_mount() {
    let mut app = demand_app();
    let view = eligible_view(&mut app);
    app.update();
    assert_eq!(views(&mut app), vec![view]);
    app.world_mut()
        .entity_mut(view)
        .remove::<TemporalAntiAliasing>();
    app.update();
    assert!(views(&mut app).is_empty());
    app.world_mut()
        .entity_mut(view)
        .insert(TemporalAntiAliasing::default());
    app.update();
    assert_eq!(views(&mut app), vec![view]);
}

/// The snapshot needs a resolved render target from the camera, not a clamped `TemporalTarget`:
/// an unresolved camera is not demand even when its `TemporalTarget` is positive.
#[test]
fn an_unresolved_target_is_not_demand_even_with_positive_temporal_target_metadata() {
    let mut app = demand_app();
    let _ = app
        .world_mut()
        .spawn((
            Camera3d::default(),
            Camera::default(),
            Msaa::Off,
            TemporalTarget::new(UVec2::new(640, 480), 1.0),
            WorldCamera,
            TemporalAntiAliasing::default(),
        ))
        .id();
    app.update();
    assert!(
        views(&mut app).is_empty(),
        "TemporalTarget::new clamps, so it cannot stand in for a resolved target"
    );
}

/// A resolved 3D world target is demand without any `TemporalTarget`.
#[test]
fn a_resolved_3d_target_is_demand_without_temporal_target_metadata() {
    let mut app = demand_app();
    let view = app
        .world_mut()
        .spawn((
            Camera3d::default(),
            resolved_camera(UVec2::new(640, 480)),
            Msaa::Off,
            WorldCamera,
            TemporalAntiAliasing::default(),
        ))
        .id();
    app.update();
    assert_eq!(views(&mut app), vec![view]);
}

/// Zero in either physical dimension is unusable, and so is an unresolved target.
#[test]
fn a_degenerate_or_unresolved_target_is_not_demand() {
    for size in [UVec2::new(0, 480), UVec2::new(640, 0), UVec2::ZERO] {
        let mut app = demand_app();
        let _ = app
            .world_mut()
            .spawn((
                Camera3d::default(),
                resolved_camera(size),
                Msaa::Off,
                TemporalTarget::new(UVec2::new(640, 480), 1.0),
                WorldCamera,
                TemporalAntiAliasing::default(),
            ))
            .id();
        app.update();
        assert!(views(&mut app).is_empty(), "{size:?} is not usable");
    }

    let mut app = demand_app();
    let _ = app
        .world_mut()
        .spawn((
            Camera3d::default(),
            Camera::default(),
            Msaa::Off,
            TemporalTarget::new(UVec2::new(640, 480), 1.0),
            WorldCamera,
            TemporalAntiAliasing::default(),
        ))
        .id();
    app.update();
    assert!(views(&mut app).is_empty(), "no resolved target");
}

/// Without `Camera3d` there is no world 3D view to snapshot, however compatible the rest is.
#[test]
fn a_non_3d_world_camera_is_not_demand() {
    let mut app = demand_app();
    let _ = app
        .world_mut()
        .spawn((
            resolved_camera(UVec2::new(640, 480)),
            Msaa::Off,
            TemporalTarget::new(UVec2::new(640, 480), 1.0),
            WorldCamera,
            TemporalAntiAliasing::default(),
        ))
        .id();
    app.update();
    assert!(views(&mut app).is_empty());
}
