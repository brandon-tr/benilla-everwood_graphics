use super::*;
use crate::static_gx::wind_source;
use bevy::camera::{CameraOutputMode, ClearColorConfig, MsaaWriteback};
use bevy::core_pipeline::core_3d::graph::Core3d;
use bevy::diagnostic::FrameCount;
use bevy::render::render_graph::RenderSubGraph;
use bevy::render::sync_world::MainEntity;
use bevy::render::view::ColorGrading;

fn rows(n: f32) -> ([f32; 4], [f32; 4]) {
    ([n, n + 1.0, n + 2.0, n + 3.0], [n + 4.0; 4])
}

fn state(frame: u32, target: UVec2, current: f32, previous: f32, valid: bool) -> GxViewWind {
    GxViewWind {
        last_frame: frame,
        target,
        buffer: None,
        current_a: rows(current).0,
        current_b: rows(current).1,
        previous_a: rows(previous).0,
        previous_b: rows(previous).1,
        valid,
    }
}

#[test]
fn first_epoch_is_invalid_and_uses_current() {
    let next = advance_wind_epoch(None, 10, UVec2::splat(64), false, rows(1.0), false);
    assert_eq!(next.current_a, rows(1.0).0);
    assert_eq!(next.current_b, rows(1.0).1);
    assert_eq!(next.previous_a, rows(1.0).0);
    assert_eq!(next.previous_b, rows(1.0).1);
    assert!(!next.valid);
    assert_eq!(next.last_frame, 10);
}

#[test]
fn consecutive_epochs_advance_previous_once() {
    let first = state(10, UVec2::splat(64), 1.0, 1.0, false);
    let second = advance_wind_epoch(Some(&first), 11, UVec2::splat(64), false, rows(2.0), false);
    assert!(second.valid);
    assert_eq!(second.current_a, rows(2.0).0);
    assert_eq!(second.current_b, rows(2.0).1);
    assert_eq!(second.previous_a, rows(1.0).0);
    assert_eq!(second.previous_b, rows(1.0).1);
    let third = advance_wind_epoch(Some(&second), 12, UVec2::splat(64), false, rows(3.0), false);
    assert!(third.valid);
    assert_eq!(third.previous_a, rows(2.0).0);
    assert_eq!(third.previous_b, rows(2.0).1);
}

#[test]
fn duplicate_frame_does_not_advance() {
    let first = state(10, UVec2::splat(64), 1.0, 1.0, false);
    let second = advance_wind_epoch(Some(&first), 11, UVec2::splat(64), false, rows(2.0), false);
    let dup = advance_wind_epoch(Some(&second), 11, UVec2::splat(64), false, rows(9.0), false);
    assert_eq!(dup.current_a, rows(2.0).0);
    assert_eq!(dup.previous_a, rows(1.0).0);
    assert!(dup.valid);
}

#[test]
fn wrapping_frame_is_consecutive() {
    let old = state(u32::MAX, UVec2::splat(64), 1.0, 1.0, true);
    let next = advance_wind_epoch(Some(&old), 0, UVec2::splat(64), false, rows(2.0), false);
    assert!(next.valid);
    assert_eq!(next.previous_a, rows(1.0).0);
}

#[test]
fn gap_reset_resize_and_buffer_change_invalidate() {
    let old = state(10, UVec2::splat(64), 1.0, 1.0, true);
    for next in [
        advance_wind_epoch(Some(&old), 12, UVec2::splat(64), false, rows(2.0), false),
        advance_wind_epoch(Some(&old), 11, UVec2::splat(128), false, rows(2.0), false),
        advance_wind_epoch(Some(&old), 11, UVec2::splat(64), true, rows(2.0), false),
        advance_wind_epoch(Some(&old), 11, UVec2::splat(64), false, rows(2.0), true),
    ] {
        assert!(!next.valid);
        assert_eq!(next.previous_a, next.current_a);
    }
}

#[test]
fn duplicate_with_caller_reset_invalidates_without_advancing() {
    let old = state(11, UVec2::splat(64), 2.0, 1.0, true);
    let next = advance_wind_epoch(Some(&old), 11, UVec2::splat(64), false, rows(9.0), true);
    assert!(!next.valid);
    assert_eq!(next.current_a, rows(2.0).0);
    assert_eq!(next.previous_a, rows(1.0).0);
}

fn view() -> (ExtractedView, ExtractedCamera) {
    let (w, h) = (64u32, 64u32);
    (
        ExtractedView {
            retained_view_entity: RetainedViewEntity::new(
                MainEntity::from(Entity::PLACEHOLDER),
                None,
                0,
            ),
            clip_from_view: Mat4::IDENTITY,
            world_from_view: GlobalTransform::IDENTITY,
            clip_from_world: None,
            hdr: true,
            viewport: UVec4::new(0, 0, w, h),
            color_grading: ColorGrading::default(),
            invert_culling: false,
        },
        ExtractedCamera {
            target: None,
            physical_viewport_size: Some(UVec2::new(w, h)),
            physical_target_size: Some(UVec2::new(w, h)),
            viewport: None,
            render_graph: Core3d.intern(),
            order: 0,
            output_mode: CameraOutputMode::Skip,
            msaa_writeback: MsaaWriteback::default(),
            clear_color: ClearColorConfig::None,
            sorted_camera_index_for_target: 0,
            exposure: 1.0,
            hdr: true,
        },
    )
}

fn epoch_app() -> (App, Entity, RetainedViewEntity) {
    let mut app = App::new();
    app.init_resource::<FrameCount>()
        .init_resource::<GxWorld>()
        .init_resource::<GxRetainedWindHistory>()
        .add_systems(Update, prepare_retained_wind_epoch);
    let (view, camera) = view();
    let key = view.retained_view_entity;
    let entity = app
        .world_mut()
        .spawn((
            view,
            camera,
            StaticGxView,
            Msaa::Off,
            TemporalAntiAliasing { reset: false },
            DepthPrepass,
            MotionVectorPrepass,
        ))
        .id();
    (app, entity, key)
}

fn run_epoch(app: &mut App, frame: u32) {
    app.world_mut().resource_mut::<FrameCount>().0 = frame;
    app.update();
}

fn with_source(app: &mut App, n: f32) {
    *app.world_mut().resource_mut::<GxWorld>() = wind_source(1);
    let mut rows = [[0.0; 4]; crate::lighting::MONKEY_FRAME_ROWS];
    rows[4] = [n, 1.0, 2.0, 3.0];
    rows[5] = [n + 4.0, 5.0, 6.0, 1.0];
    app.world_mut()
        .insert_resource(GxRetainedWindFrame::from_rows(&rows));
}

fn establish_valid_epoch(app: &mut App, entity: Entity, key: RetainedViewEntity, n: f32) {
    with_source(app, n);
    run_epoch(app, 1);
    app.world_mut()
        .entity_mut(entity)
        .insert(TemporalAntiAliasing { reset: false });
    run_epoch(app, 2);
    let history = app.world().resource::<GxRetainedWindHistory>();
    let state = history.views.get(&key).expect("valid epoch history");
    assert!(state.valid);
    let prepared = app.world().entity(entity).get::<GxPreparedWind>().unwrap();
    assert_eq!(prepared.data.flags.x, 1.0);
    assert_eq!(prepared.data.flags.y, 1.0);
    assert!(prepared.wind_ready);
}

#[test]
fn ecs_taa_removal_retires_and_reactivation_resets() {
    let (mut app, e, key) = epoch_app();
    establish_valid_epoch(&mut app, e, key, 1.0);
    app.world_mut()
        .entity_mut(e)
        .insert(GxPreviousWindOffset(123));
    app.world_mut()
        .entity_mut(e)
        .remove::<TemporalAntiAliasing>();
    run_epoch(&mut app, 3);
    let entity = app.world().entity(e);
    assert!(!entity.contains::<GxPreparedWind>());
    assert!(!entity.contains::<GxPreviousWindOffset>());
    assert!(app
        .world()
        .resource::<GxRetainedWindHistory>()
        .views
        .is_empty());
    app.world_mut()
        .entity_mut(e)
        .insert(TemporalAntiAliasing { reset: false });
    run_epoch(&mut app, 4);
    let entity = app.world().entity(e);
    assert!(entity.get::<TemporalAntiAliasing>().unwrap().reset);
    assert_eq!(entity.get::<GxPreparedWind>().unwrap().data.flags.x, 0.0);
    let state = app
        .world()
        .resource::<GxRetainedWindHistory>()
        .views
        .get(&key)
        .copied()
        .expect("reactivated history");
    assert!(!state.valid);
    assert_eq!(state.previous_a, state.current_a);
    assert_eq!(state.previous_b, state.current_b);
}

#[test]
fn ecs_required_component_loss_retires_state() {
    for remove in ["depth", "motion", "camera", "view"] {
        let (mut app, e, key) = epoch_app();
        establish_valid_epoch(&mut app, e, key, 1.0);
        app.world_mut()
            .entity_mut(e)
            .insert(GxPreviousWindOffset(123));
        let mut entity = app.world_mut().entity_mut(e);
        match remove {
            "depth" => {
                entity.remove::<DepthPrepass>();
            }
            "motion" => {
                entity.remove::<MotionVectorPrepass>();
            }
            "camera" => {
                entity.remove::<ExtractedCamera>();
            }
            "view" => {
                entity.remove::<ExtractedView>();
            }
            _ => unreachable!(),
        }
        run_epoch(&mut app, 2);
        let entity = app.world().entity(e);
        assert!(!entity.contains::<GxPreparedWind>(), "{remove}");
        assert!(!entity.contains::<GxPreviousWindOffset>(), "{remove}");
        assert!(
            app.world()
                .resource::<GxRetainedWindHistory>()
                .views
                .is_empty(),
            "{remove}"
        );
    }
}

#[test]
fn ecs_unsupported_views_retire_state() {
    for case in ["msaa", "deferred", "viewport", "override", "zero", "mirror"] {
        let (mut app, e, key) = epoch_app();
        establish_valid_epoch(&mut app, e, key, 1.0);
        app.world_mut()
            .entity_mut(e)
            .insert(GxPreviousWindOffset(123));
        let mut entity = app.world_mut().entity_mut(e);
        match case {
            "msaa" => {
                entity.insert(Msaa::Sample4);
            }
            "deferred" => {
                entity.insert(DeferredPrepass);
            }
            "viewport" => {
                if let Some(mut camera) = entity.get_mut::<ExtractedCamera>() {
                    camera.viewport = Some(bevy::camera::Viewport {
                        physical_position: UVec2::ZERO,
                        physical_size: UVec2::splat(16),
                        ..Default::default()
                    });
                }
            }
            "override" => {
                entity.insert(bevy::camera::MainPassResolutionOverride(UVec2::splat(16)));
            }
            "zero" => {
                if let Some(mut camera) = entity.get_mut::<ExtractedCamera>() {
                    camera.physical_target_size = Some(UVec2::ZERO);
                }
            }
            "mirror" => {
                entity.insert(StaticGxMirror);
            }
            _ => unreachable!(),
        }
        run_epoch(&mut app, 2);
        let entity = app.world().entity(e);
        assert!(!entity.contains::<GxPreparedWind>(), "{case}");
        assert!(!entity.contains::<GxPreviousWindOffset>(), "{case}");
        assert!(
            app.world()
                .resource::<GxRetainedWindHistory>()
                .views
                .is_empty(),
            "{case}"
        );
    }
}

#[test]
fn ecs_source_loss_and_recovery() {
    let (mut app, e, key) = epoch_app();
    establish_valid_epoch(&mut app, e, key, 1.0);
    app.world_mut()
        .entity_mut(e)
        .insert(GxPreviousWindOffset(77));
    assert!(app.world().entity(e).contains::<GxPreviousWindOffset>());
    app.world_mut().remove_resource::<GxRetainedWindFrame>();
    app.world_mut()
        .entity_mut(e)
        .insert(TemporalAntiAliasing { reset: false });
    run_epoch(&mut app, 3);
    assert!(app
        .world()
        .resource::<GxRetainedWindHistory>()
        .views
        .is_empty());
    let entity = app.world().entity(e);
    let prepared = entity.get::<GxPreparedWind>().unwrap();
    assert!(!prepared.wind_ready);
    assert_eq!(prepared.data.flags.x, 0.0);
    assert_eq!(prepared.data.flags.y, 0.0);
    assert!(!entity.contains::<GxPreviousWindOffset>());
    assert!(entity.get::<TemporalAntiAliasing>().unwrap().reset);

    with_source(&mut app, 2.0);
    app.world_mut()
        .entity_mut(e)
        .insert(TemporalAntiAliasing { reset: false });
    run_epoch(&mut app, 4);
    let entity = app.world().entity(e);
    let prepared = entity.get::<GxPreparedWind>().unwrap();
    assert_eq!(prepared.data.flags.x, 0.0);
    assert_eq!(prepared.data.flags.y, 1.0);
    assert!(prepared.wind_ready);
    assert!(entity.get::<TemporalAntiAliasing>().unwrap().reset);
    let recovered = *app
        .world()
        .resource::<GxRetainedWindHistory>()
        .views
        .get(&key)
        .expect("source recovery history");
    assert!(!recovered.valid);
    assert_eq!(recovered.previous_a, recovered.current_a);
    assert_eq!(recovered.previous_b, recovered.current_b);
    let current_a = recovered.current_a;
    let current_b = recovered.current_b;

    app.world_mut()
        .entity_mut(e)
        .insert(TemporalAntiAliasing { reset: false });
    run_epoch(&mut app, 5);
    let prepared = app.world().entity(e).get::<GxPreparedWind>().unwrap();
    assert_eq!(prepared.data.flags.x, 1.0);
    let next = *app
        .world()
        .resource::<GxRetainedWindHistory>()
        .views
        .get(&key)
        .expect("consecutive recovery history");
    assert!(next.valid);
    assert_eq!(next.previous_a, current_a);
    assert_eq!(next.previous_b, current_b);
}

#[test]
fn ecs_rigid_missing_source_does_not_force_reset() {
    let (mut app, e, _) = epoch_app();
    run_epoch(&mut app, 1);
    let entity = app.world().entity(e);
    assert!(entity.get::<GxPreparedWind>().is_some());
    assert!(!entity.get::<GxPreparedWind>().unwrap().wind_ready);
    assert!(app
        .world()
        .resource::<GxRetainedWindHistory>()
        .views
        .is_empty());
    assert!(!entity.get::<TemporalAntiAliasing>().unwrap().reset);
}

#[test]
fn ecs_caller_reset_is_preserved() {
    let (mut app, e, _) = epoch_app();
    app.world_mut()
        .entity_mut(e)
        .insert(TemporalAntiAliasing { reset: true });
    run_epoch(&mut app, 1);
    assert!(
        app.world()
            .entity(e)
            .get::<TemporalAntiAliasing>()
            .unwrap()
            .reset
    );
}
