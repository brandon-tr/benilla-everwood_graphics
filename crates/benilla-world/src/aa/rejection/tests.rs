use super::*;
use crate::aa::skin_bounds::SkinBoundsError;
use crate::aa::{AaEffective, AaMode, AaPolicy, AaReason, TemporalHistory, TemporalTarget};
use crate::view::world_fovy;
use bevy::ecs::schedule::Schedule;
use bevy::render::{ExtractSchedule, MainWorld};

fn key() -> DepthKey {
    DepthKey {
        main_view: Entity::from_raw_u32(1).unwrap(),
        generation: 3,
        size: UVec2::new(200, 100),
    }
}

fn clip() -> Mat4 {
    Mat4::perspective_infinite_reverse_rh(world_fovy(2.0), 2.0, 1.0)
}

fn bounds() -> Bounds3 {
    Bounds3 {
        min: Vec3::new(-1.0, -1.0, -6.0),
        max: Vec3::new(1.0, 1.0, -4.0),
    }
}

fn extraction_fixture(effective: bool) -> (World, Schedule, Entity, Entity) {
    let mut render = World::new();
    let mut main = MainWorld::default();
    main.insert_resource(Assets::<WowModelMaterial>::default());
    let mut policy = AaPolicy::default();
    if effective {
        policy.request(AaMode::Taa);
        policy.effective = AaEffective::Taa;
        policy.reason = AaReason::TaaRequested;
    }
    main.insert_resource(policy);
    let target = TemporalTarget::new(UVec2::new(200, 100), 1.0);
    let camera = main.spawn_empty().id();
    let render_key = loop {
        let candidate = render.spawn_empty().id();
        if candidate != camera {
            break candidate;
        }
    };
    assert_ne!(camera, render_key);
    let mut camera_component = Camera::default();
    camera_component.computed.clip_from_view = clip();
    main.entity_mut(camera).insert((
        WorldCamera,
        camera_component,
        Msaa::Off,
        RenderEntity::from(render_key),
        VisibleEntities::default(),
        target,
        GlobalTransform::IDENTITY,
    ));
    let mut history = TemporalHistory::default();
    if effective {
        history.register(camera, target);
    }
    main.insert_resource(history);

    render.insert_resource(main);
    render.insert_resource(TemporalDepthRequests::default());
    render.insert_resource(crate::aa::native::NativeResolveBridge::default());
    render.insert_resource(ModelRejectRequests::default());
    let mut schedule = Schedule::new(ExtractSchedule);
    schedule.add_systems((depth::extract_depth_requests, extract_model_rejections).chain());
    (render, schedule, camera, render_key)
}

fn two_camera_fixture() -> (World, Schedule, Entity, Entity, Entity, Entity) {
    let mut render = World::new();
    let mut main = MainWorld::default();
    main.insert_resource(Assets::<WowModelMaterial>::default());
    let mut policy = AaPolicy::default();
    policy.request(AaMode::Taa);
    policy.effective = AaEffective::Taa;
    policy.reason = AaReason::TaaRequested;
    main.insert_resource(policy);

    let target_a = TemporalTarget::new(UVec2::new(200, 100), 1.0);
    let target_b = TemporalTarget::new(UVec2::new(400, 200), 1.0);
    let camera_a = main.spawn_empty().id();
    let camera_b = main.spawn_empty().id();
    let render_a = loop {
        let candidate = render.spawn_empty().id();
        if candidate != camera_a && candidate != camera_b {
            break candidate;
        }
    };
    let render_b = loop {
        let candidate = render.spawn_empty().id();
        if candidate != camera_a && candidate != camera_b && candidate != render_a {
            break candidate;
        }
    };
    assert!(camera_a != camera_b);
    assert!(render_a != render_b);
    assert!(render_a != camera_a && render_a != camera_b);
    assert!(render_b != camera_a && render_b != camera_b);

    for (camera, render_key, target) in [
        (camera_a, render_a, target_a),
        (camera_b, render_b, target_b),
    ] {
        let mut camera_component = Camera::default();
        camera_component.computed.clip_from_view = clip();
        main.entity_mut(camera).insert((
            WorldCamera,
            camera_component,
            Msaa::Off,
            RenderEntity::from(render_key),
            VisibleEntities::default(),
            target,
            GlobalTransform::IDENTITY,
        ));
    }
    let mut history = TemporalHistory::default();
    history.register(camera_a, target_a);
    history.register(camera_b, target_b);
    main.insert_resource(history);

    render.insert_resource(main);
    render.insert_resource(TemporalDepthRequests::default());
    render.insert_resource(crate::aa::native::NativeResolveBridge::default());
    render.insert_resource(ModelRejectRequests::default());
    let mut schedule = Schedule::new(ExtractSchedule);
    schedule.add_systems((depth::extract_depth_requests, extract_model_rejections).chain());
    (render, schedule, camera_a, camera_b, render_a, render_b)
}

#[test]
fn extraction_uses_distinct_main_and_render_entities() {
    let (mut render, mut schedule, main_camera, render_key) = extraction_fixture(true);
    assert_ne!(main_camera, render_key);
    let (mapped, generation) = {
        let main = render.resource::<MainWorld>();
        let mapped = main
            .get::<RenderEntity>(main_camera)
            .expect("explicit render mapping")
            .id();
        let generation = main
            .resource::<TemporalHistory>()
            .view(main_camera)
            .expect("registered history")
            .generation();
        (mapped, generation)
    };
    assert_eq!(mapped, render_key);
    assert!(
        !render
            .resource::<MainWorld>()
            .contains_resource::<TemporalDepthRequests>(),
        "depth requests must live only in the render world"
    );

    schedule.run(&mut render);
    assert!(render.get_entity(render_key).is_ok());
    let depth = render.resource::<TemporalDepthRequests>();
    let request = depth.get(render_key).expect("render-keyed depth request");
    assert_eq!(request.key().main_view, main_camera);
    assert_eq!(request.key().size, UVec2::new(200, 100));
    assert_eq!(request.key().generation, generation);
    let rejection = render.resource::<ModelRejectRequests>();
    assert_eq!(rejection.by_render.len(), 1);
    assert!(!rejection.by_render.contains_key(&main_camera));
    let view = rejection
        .by_render
        .get(&render_key)
        .expect("render-keyed rejection plan");
    assert_eq!(view.key, request.key());
    assert!(!view.full_view);
    assert!(view.rects.is_empty());
}

#[test]
fn extraction_isolates_two_eligible_views_across_frames() {
    let (mut render, mut schedule, camera_a, camera_b, render_a, render_b) = two_camera_fixture();
    let expected_a = {
        let main = render.resource::<MainWorld>();
        DepthKey {
            main_view: camera_a,
            generation: main
                .resource::<TemporalHistory>()
                .view(camera_a)
                .unwrap()
                .generation(),
            size: UVec2::new(200, 100),
        }
    };
    let expected_b = {
        let main = render.resource::<MainWorld>();
        DepthKey {
            main_view: camera_b,
            generation: main
                .resource::<TemporalHistory>()
                .view(camera_b)
                .unwrap()
                .generation(),
            size: UVec2::new(400, 200),
        }
    };

    // Frame 1: both eligible, both clear.
    schedule.run(&mut render);
    let depth_epoch = render.resource::<TemporalDepthRequests>().epoch();
    assert_eq!(depth_epoch, 1);
    assert_eq!(render.resource::<TemporalDepthRequests>().iter().count(), 2);
    let rejection = render.resource::<ModelRejectRequests>();
    assert_eq!(rejection.epoch, depth_epoch);
    assert_eq!(rejection.by_render.len(), 2);
    for (key, expected) in [(render_a, expected_a), (render_b, expected_b)] {
        let view = rejection.by_render.get(&key).expect("independent plan");
        assert_eq!(view.key, expected);
        assert!(!view.full_view);
        assert!(view.rects.is_empty());
    }

    // Frame 2: only A's visible list becomes inconsistent.
    let missing = {
        let mut main = render.resource_mut::<MainWorld>();
        let missing = main.spawn_empty().id();
        main.get_mut::<VisibleEntities>(camera_a)
            .unwrap()
            .push(missing, TypeId::of::<Mesh3d>());
        missing
    };
    schedule.run(&mut render);
    let depth_epoch = render.resource::<TemporalDepthRequests>().epoch();
    assert_eq!(depth_epoch, 2);
    let rejection = render.resource::<ModelRejectRequests>();
    assert_eq!(rejection.epoch, depth_epoch);
    assert_eq!(rejection.by_render.len(), 2);
    assert!(rejection.by_render.get(&render_a).unwrap().full_view);
    assert!(!rejection.by_render.get(&render_b).unwrap().full_view);
    assert!(rejection.by_render.get(&render_b).unwrap().rects.is_empty());
    assert_eq!(rejection.by_render.get(&render_a).unwrap().key, expected_a);
    assert_eq!(rejection.by_render.get(&render_b).unwrap().key, expected_b);

    // Frame 3: the inconsistency moves from A to B; FullView must not leak.
    {
        let mut main = render.resource_mut::<MainWorld>();
        main.get_mut::<VisibleEntities>(camera_a)
            .unwrap()
            .clear(TypeId::of::<Mesh3d>());
        main.get_mut::<VisibleEntities>(camera_b)
            .unwrap()
            .push(missing, TypeId::of::<Mesh3d>());
    }
    schedule.run(&mut render);
    let depth_epoch = render.resource::<TemporalDepthRequests>().epoch();
    assert_eq!(depth_epoch, 3);
    let rejection = render.resource::<ModelRejectRequests>();
    assert_eq!(rejection.epoch, depth_epoch);
    assert_eq!(rejection.by_render.len(), 2);
    assert!(!rejection.by_render.get(&render_a).unwrap().full_view);
    assert!(rejection.by_render.get(&render_b).unwrap().full_view);
    assert!(rejection.by_render.get(&render_a).unwrap().rects.is_empty());
    assert!(rejection.by_render.get(&render_b).unwrap().rects.is_empty());

    // Frame 4: retire A; B keeps its own plan and its FullView state.
    *render
        .resource_mut::<MainWorld>()
        .get_mut::<Msaa>(camera_a)
        .unwrap() = Msaa::Sample4;
    schedule.run(&mut render);
    let depth_epoch = render.resource::<TemporalDepthRequests>().epoch();
    assert_eq!(depth_epoch, 4);
    assert_eq!(render.resource::<TemporalDepthRequests>().iter().count(), 1);
    assert!(render
        .resource::<TemporalDepthRequests>()
        .get(render_a)
        .is_none());
    let rejection = render.resource::<ModelRejectRequests>();
    assert_eq!(rejection.epoch, depth_epoch);
    assert_eq!(rejection.by_render.len(), 1);
    assert!(!rejection.by_render.contains_key(&render_a));
    let b_plan = rejection.by_render.get(&render_b).expect("B remains");
    assert_eq!(b_plan.key, expected_b);
    assert!(b_plan.full_view);

    // Frame 5: retire B too; both maps empty and still share the depth epoch.
    *render
        .resource_mut::<MainWorld>()
        .get_mut::<Msaa>(camera_b)
        .unwrap() = Msaa::Sample4;
    schedule.run(&mut render);
    let depth = render.resource::<TemporalDepthRequests>();
    assert_eq!(depth.epoch(), 5);
    assert_eq!(depth.iter().count(), 0);
    let rejection = render.resource::<ModelRejectRequests>();
    assert_eq!(rejection.epoch, depth.epoch());
    assert!(rejection.by_render.is_empty());
}

#[test]
fn extraction_reads_render_world_depth_requests() {
    let (mut render, mut schedule, _camera, render_key) = extraction_fixture(true);
    assert!(
        !render
            .resource::<MainWorld>()
            .contains_resource::<TemporalDepthRequests>(),
        "depth requests must live only in the render world"
    );
    schedule.run(&mut render);
    let depth = render.resource::<TemporalDepthRequests>();
    assert_eq!(depth.epoch(), 1);
    let request = depth.get(render_key).expect("eligible camera request");
    let rejection = render.resource::<ModelRejectRequests>();
    assert_eq!(rejection.epoch, depth.epoch());
    let view = rejection
        .by_render
        .get(&render_key)
        .expect("render-keyed rejection plan");
    assert_eq!(view.key, request.key());
    assert!(!view.full_view);
    assert!(view.rects.is_empty());

    // Removing eligibility must remove the old plan and follow the next depth epoch.
    render
        .resource_mut::<MainWorld>()
        .resource_mut::<AaPolicy>()
        .effective = AaEffective::Msaa { samples: 1 };
    schedule.run(&mut render);
    assert_eq!(render.resource::<TemporalDepthRequests>().epoch(), 2);
    assert!(render
        .resource::<TemporalDepthRequests>()
        .get(render_key)
        .is_none());
    assert!(render
        .resource::<ModelRejectRequests>()
        .by_render
        .is_empty());
}

#[test]
fn extraction_fallback_has_no_plans() {
    let (mut render, mut schedule, _camera, _render_key) = extraction_fixture(false);
    schedule.run(&mut render);
    assert_eq!(render.resource::<TemporalDepthRequests>().epoch(), 1);
    assert_eq!(render.resource::<ModelRejectRequests>().epoch, 1);
    assert!(render
        .resource::<ModelRejectRequests>()
        .by_render
        .is_empty());
}

#[test]
fn extraction_rejects_missing_mesh_data_in_visible_list() {
    let (mut render, mut schedule, camera, render_key) = extraction_fixture(true);
    let missing = {
        let mut main = render.resource_mut::<MainWorld>();
        let missing = main.spawn_empty().id();
        main.get_mut::<VisibleEntities>(camera)
            .unwrap()
            .push(missing, TypeId::of::<Mesh3d>());
        missing
    };
    schedule.run(&mut render);
    let view = render
        .resource::<ModelRejectRequests>()
        .by_render
        .get(&render_key)
        .expect("rejection plan");
    assert!(view.full_view, "missing mesh data must make FullView");
    assert!(view.rects.is_empty());
    let _ = missing;

    let (mut render, mut schedule, camera, render_key) = extraction_fixture(true);
    {
        let mut main = render.resource_mut::<MainWorld>();
        let missing = main.spawn_empty().id();
        main.get_mut::<VisibleEntities>(camera)
            .unwrap()
            .push(missing, TypeId::of::<Mesh3d>());
        main.despawn(missing);
    }
    schedule.run(&mut render);
    let view = render
        .resource::<ModelRejectRequests>()
        .by_render
        .get(&render_key)
        .expect("rejection plan");
    assert!(view.full_view, "despawned listed mesh must make FullView");
    assert!(view.rects.is_empty());
}

#[test]
fn extraction_keeps_camera_lists_separate_and_excludes_other_materials() {
    let (mut render, mut schedule, camera, render_key) = extraction_fixture(true);
    let other_render = loop {
        let candidate = render.spawn_empty().id();
        if candidate != camera && candidate != render_key {
            break candidate;
        }
    };
    {
        let mut main = render.resource_mut::<MainWorld>();
        let other = main.spawn_empty().id();
        let mut other_camera = Camera::default();
        other_camera.computed.clip_from_view = clip();
        let mut other_visible = VisibleEntities::default();
        other_visible.push(other, TypeId::of::<Mesh3d>());
        main.entity_mut(other).insert((
            WorldCamera,
            other_camera,
            Msaa::Off,
            RenderEntity::from(other_render),
            other_visible,
            GlobalTransform::IDENTITY,
        ));
        let layered = main
            .spawn((
                Mesh3d(Handle::<Mesh>::default()),
                MeshMaterial3d(Handle::<WowModelMaterial>::default()),
                MeshTag(crate::mesh_tag::spawn_tag(1, 1.0)),
                RenderLayers::layer(1),
            ))
            .id();
        main.get_mut::<VisibleEntities>(camera)
            .unwrap()
            .push(layered, TypeId::of::<Mesh3d>());
    }
    schedule.run(&mut render);
    let requested = render
        .resource::<ModelRejectRequests>()
        .by_render
        .get(&render_key)
        .expect("requested view plan");
    assert!(!requested.full_view);
    assert!(requested.rects.is_empty());

    // A successful Mesh3d query without WowModelMaterial stays excluded.
    let (mut render, mut schedule, camera, render_key) = extraction_fixture(true);
    {
        let mut main = render.resource_mut::<MainWorld>();
        let plain = main.spawn(Mesh3d(Handle::<Mesh>::default())).id();
        main.get_mut::<VisibleEntities>(camera)
            .unwrap()
            .push(plain, TypeId::of::<Mesh3d>());
    }
    schedule.run(&mut render);
    let requested = render
        .resource::<ModelRejectRequests>()
        .by_render
        .get(&render_key)
        .expect("requested view plan");
    assert!(!requested.full_view);
    assert!(requested.rects.is_empty());

    // An in-layer WowModelMaterial model with no material asset is unsafe, not skipped.
    let (mut render, mut schedule, camera, render_key) = extraction_fixture(true);
    {
        let mut main = render.resource_mut::<MainWorld>();
        let missing_material = main
            .spawn((
                Mesh3d(Handle::<Mesh>::default()),
                MeshMaterial3d(Handle::<WowModelMaterial>::default()),
                MeshTag(crate::mesh_tag::spawn_tag(1, 1.0)),
                RenderLayers::layer(0),
            ))
            .id();
        main.get_mut::<VisibleEntities>(camera)
            .unwrap()
            .push(missing_material, TypeId::of::<Mesh3d>());
    }
    schedule.run(&mut render);
    let requested = render
        .resource::<ModelRejectRequests>()
        .by_render
        .get(&render_key)
        .expect("requested view plan");
    assert!(requested.full_view);
    assert!(requested.rects.is_empty());
}

fn retained_frame() -> MonkeyFrame {
    MonkeyFrame {
        wind_dir: [1.0, 0.0],
        ..Default::default()
    }
}

fn add_retained(render: &mut World, count: usize, with_frame: bool) {
    let mut main = render.resource_mut::<MainWorld>();
    main.insert_resource(crate::static_gx::wind_source(count));
    if with_frame {
        main.insert_resource(retained_frame());
    }
}

#[test]
#[allow(clippy::reversed_empty_ranges)] // deliberately malformed published range
fn retained_malformed_draw_after_wind_rejects_extracted_view() {
    let (mut render, mut schedule, _camera, render_key) = extraction_fixture(true);
    let mut source = crate::static_gx::wind_source(1);
    let cell = (0, 0);
    let mut region = (*source.cells[&cell]).clone();
    let mut malformed = region.draws[0].clone();
    malformed.index_range = 5..2;
    region.draws.push(malformed);
    source.cells.insert(cell, std::sync::Arc::new(region));
    {
        let mut main = render.resource_mut::<MainWorld>();
        main.insert_resource(source);
        main.insert_resource(retained_frame());
    }
    schedule.run(&mut render);
    let view = render
        .resource::<ModelRejectRequests>()
        .by_render
        .get(&render_key)
        .expect("rejection plan");
    assert!(view.full_view);
    assert!(view.rects.is_empty());
}

#[test]
fn retained_wind_adds_a_rectangle_to_an_empty_entity_view() {
    let (mut render, mut schedule, _camera, render_key) = extraction_fixture(true);
    add_retained(&mut render, 1, true);
    schedule.run(&mut render);
    let view = render
        .resource::<ModelRejectRequests>()
        .by_render
        .get(&render_key)
        .expect("rejection plan");
    assert!(!view.full_view);
    assert_eq!(view.rects.len(), 1);
}

#[test]
fn retained_wind_without_frame_or_with_cap_overflow_rejects_full_view() {
    let (mut render, mut schedule, _camera, render_key) = extraction_fixture(true);
    add_retained(&mut render, 1, false);
    schedule.run(&mut render);
    let view = render
        .resource::<ModelRejectRequests>()
        .by_render
        .get(&render_key)
        .expect("rejection plan");
    assert!(view.full_view);
    assert!(view.rects.is_empty());

    let (mut render, mut schedule, _camera, render_key) = extraction_fixture(true);
    add_retained(&mut render, MAX_RECTS_PER_VIEW + 1, true);
    schedule.run(&mut render);
    let view = render
        .resource::<ModelRejectRequests>()
        .by_render
        .get(&render_key)
        .expect("rejection plan");
    assert!(view.full_view);
    assert!(view.rects.is_empty());
}

#[test]
fn retained_render_fallback_still_contributes() {
    let (mut render, mut schedule, _camera, render_key) = extraction_fixture(true);
    render.insert_resource(crate::static_gx::wind_source(1));
    render
        .resource_mut::<MainWorld>()
        .insert_resource(retained_frame());
    schedule.run(&mut render);
    let view = render
        .resource::<ModelRejectRequests>()
        .by_render
        .get(&render_key)
        .expect("rejection plan");
    assert!(!view.full_view);
    assert_eq!(view.rects.len(), 1);
}

#[test]
fn retained_wind_projects_independently_for_two_cameras() {
    let (mut render, mut schedule, _camera_a, camera_b, render_a, render_b) = two_camera_fixture();
    {
        let mut main = render.resource_mut::<MainWorld>();
        main.insert_resource(crate::static_gx::wind_source(1));
        main.insert_resource(retained_frame());
        *main.get_mut::<GlobalTransform>(camera_b).unwrap() =
            GlobalTransform::from_translation(Vec3::new(1000.0, 0.0, 0.0));
    }
    schedule.run(&mut render);
    let requests = render.resource::<ModelRejectRequests>();
    let a = requests.by_render.get(&render_a).expect("camera A plan");
    assert!(!a.full_view);
    assert_eq!(a.rects.len(), 1);
    let b = requests.by_render.get(&render_b).expect("camera B plan");
    assert!(!b.full_view);
    assert!(b.rects.is_empty());
}

#[test]
fn no_retained_source_keeps_the_empty_scene_clear() {
    let (mut render, mut schedule, _camera, render_key) = extraction_fixture(true);
    schedule.run(&mut render);
    let view = render
        .resource::<ModelRejectRequests>()
        .by_render
        .get(&render_key)
        .expect("rejection plan");
    assert!(!view.full_view);
    assert!(view.rects.is_empty());
}

#[test]
fn centered_box_projects_to_a_half_open_rect() {
    let projected = project_reject_bounds(
        Vec3::ZERO,
        bounds(),
        Mat4::IDENTITY,
        clip(),
        UVec2::new(200, 100),
    );
    let ProjectedReject::Rect(rect) = projected else {
        panic!("expected rect, got {projected:?}");
    };
    assert!(rect.min.x < rect.max.x);
    assert!(rect.min.y < rect.max.y);
    assert!(rect.min.x >= 2);
    assert!(rect.min.y >= 2);
    assert!(rect.max.x <= 198);
    assert!(rect.max.y <= 98);
}

#[test]
fn partial_offscreen_is_clipped_and_fully_outside_is_empty() {
    let mut partial = bounds();
    partial.min.x = 4.0;
    partial.max.x = 6.0;
    let projected = project_reject_bounds(
        Vec3::ZERO,
        partial,
        Mat4::IDENTITY,
        clip(),
        UVec2::new(200, 100),
    );
    assert!(matches!(projected, ProjectedReject::Rect(_)));

    let mut outside = bounds();
    outside.min.x = 1000.0;
    outside.max.x = 1001.0;
    assert_eq!(
        project_reject_bounds(
            Vec3::ZERO,
            outside,
            Mat4::IDENTITY,
            clip(),
            UVec2::new(200, 100)
        ),
        ProjectedReject::Outside
    );
}

#[test]
fn near_plane_and_invalid_inputs_reject_full_view() {
    let mut crossing = bounds();
    crossing.min.z = -2.0;
    crossing.max.z = 2.0;
    assert_eq!(
        project_reject_bounds(
            Vec3::ZERO,
            crossing,
            Mat4::IDENTITY,
            clip(),
            UVec2::new(200, 100)
        ),
        ProjectedReject::FullView
    );
    assert_eq!(
        project_reject_bounds(
            Vec3::NAN,
            bounds(),
            Mat4::IDENTITY,
            clip(),
            UVec2::new(200, 100)
        ),
        ProjectedReject::FullView
    );
    let mut unordered = bounds();
    unordered.min.x = 10.0;
    assert_eq!(
        project_reject_bounds(
            Vec3::ZERO,
            unordered,
            Mat4::IDENTITY,
            clip(),
            UVec2::new(200, 100)
        ),
        ProjectedReject::FullView
    );
    assert_eq!(
        project_reject_bounds(
            Vec3::ZERO,
            bounds(),
            Mat4::ZERO,
            clip(),
            UVec2::new(200, 100)
        ),
        ProjectedReject::FullView
    );
    assert_eq!(
        project_reject_bounds(
            Vec3::ZERO,
            bounds(),
            Mat4::IDENTITY,
            Mat4::ZERO,
            UVec2::new(200, 100)
        ),
        ProjectedReject::FullView
    );
    assert_eq!(
        project_reject_bounds(Vec3::ZERO, bounds(), Mat4::IDENTITY, clip(), UVec2::ZERO),
        ProjectedReject::FullView
    );
}

#[test]
fn large_world_offset_preserves_the_rectangle() {
    let extent = UVec2::new(200, 100);
    let base = project_reject_bounds(Vec3::ZERO, bounds(), Mat4::IDENTITY, clip(), extent);
    let offset = Vec3::new(9000.0, 0.0, 0.0);
    let shifted = project_reject_bounds(
        offset,
        bounds(),
        Mat4::from_translation(offset),
        clip(),
        extent,
    );
    assert_eq!(base, shifted);
}

#[test]
fn negative_scale_rigid_aabb_still_rejects() {
    let aabb = Aabb::from_min_max(Vec3::splat(-1.0), Vec3::splat(1.0));
    let transform = GlobalTransform::from(Mat4::from_scale_rotation_translation(
        Vec3::new(-1.0, 1.0, 1.0),
        Quat::from_rotation_y(0.4),
        Vec3::new(0.0, 0.0, -5.0),
    ));
    let projected = project_rigid_aabb(
        &aabb,
        &transform,
        Mat4::IDENTITY,
        clip(),
        UVec2::new(200, 100),
    );
    assert!(matches!(projected, ProjectedReject::Rect(_)));
}

#[test]
fn rect_cap_becomes_full_view_not_a_truncated_mask() {
    let mut view = ViewModelRejects::new(key());
    let rect = URect {
        min: UVec2::new(1, 1),
        max: UVec2::new(2, 2),
    };
    for _ in 0..MAX_RECTS_PER_VIEW {
        assert!(push_rect(&mut view, rect));
    }
    assert!(!push_rect(&mut view, rect));
    assert!(view.full_view);
    assert!(view.rects.is_empty());
}

#[test]
fn layers_and_material_markers_are_conservative() {
    assert!(layer_visible(None, None));
    assert!(layer_visible(Some(&RenderLayers::layer(0)), None));
    assert!(!layer_visible(
        Some(&RenderLayers::layer(1)),
        Some(&RenderLayers::layer(0))
    ));
    assert!(layer_visible(
        Some(&RenderLayers::layer(0)),
        Some(&RenderLayers::layer(0))
    ));

    let mut fade = Vec4::ZERO;
    assert_eq!(marker_disposition(fade), MaterialDisposition::Clear);
    fade.w = 1.0;
    assert_eq!(marker_disposition(fade), MaterialDisposition::FullView);
    fade.w = 0.0;
    fade.z = f32::from(FOLIAGE_WIND_MARKER);
    assert_eq!(marker_disposition(fade), MaterialDisposition::FullView);
    fade.z = f32::from(SKY_DEPTH_MARKER);
    assert_eq!(marker_disposition(fade), MaterialDisposition::SkyOnly);
    fade.z = f32::NAN;
    assert_eq!(marker_disposition(fade), MaterialDisposition::FullView);
}

#[test]
fn posed_metadata_is_used_instead_of_an_unposed_aabb() {
    let mesh = bevy::asset::AssetId::Uuid {
        uuid: bevy::asset::uuid::Uuid::from_u128(4),
    };
    let skinned = TemporalSkinBounds {
        mesh,
        slot: 1,
        result: SkinBoundsResult::Skinned {
            rig_origin: Vec3::ZERO,
            relative_bounds: bounds(),
        },
    };
    assert!(metadata_consistent(&skinned, mesh, 1));
    assert!(!metadata_consistent(&skinned, mesh, 2));
    let far_aabb = Aabb::from_min_max(Vec3::splat(-1000.0), Vec3::splat(1000.0));
    let projected = project_model_metadata(
        &skinned,
        Some(&far_aabb),
        None,
        Mat4::IDENTITY,
        clip(),
        UVec2::new(200, 100),
    );
    assert!(
        matches!(projected, ProjectedReject::Rect(_)),
        "posed bounds must win over the mesh AABB: {projected:?}"
    );
}

#[test]
fn not_skinned_uses_transform_and_missing_data_is_full_view() {
    let mesh = bevy::asset::AssetId::Uuid {
        uuid: bevy::asset::uuid::Uuid::from_u128(5),
    };
    let not_skinned = TemporalSkinBounds {
        mesh,
        slot: 0,
        result: SkinBoundsResult::NotSkinned,
    };
    let aabb = Aabb::from_min_max(Vec3::splat(-1.0), Vec3::splat(1.0));
    let transform = GlobalTransform::from_translation(Vec3::new(0.0, 0.0, -5.0));
    let projected = project_model_metadata(
        &not_skinned,
        Some(&aabb),
        Some(&transform),
        Mat4::IDENTITY,
        clip(),
        UVec2::new(200, 100),
    );
    assert!(matches!(projected, ProjectedReject::Rect(_)));
    assert_eq!(
        project_model_metadata(
            &not_skinned,
            None,
            Some(&transform),
            Mat4::IDENTITY,
            clip(),
            UVec2::new(200, 100)
        ),
        ProjectedReject::FullView
    );
    let unsafe_bounds = TemporalSkinBounds {
        mesh,
        slot: 0,
        result: SkinBoundsResult::Unsafe(SkinBoundsError::InvalidPalette),
    };
    assert_eq!(
        project_model_metadata(
            &unsafe_bounds,
            Some(&aabb),
            Some(&transform),
            Mat4::IDENTITY,
            clip(),
            UVec2::new(200, 100)
        ),
        ProjectedReject::FullView
    );
}

#[test]
fn output_lifecycle_replaces_stale_generation_and_retires_missing_depth() {
    let requested = key();
    assert_eq!(
        reject_output_decision(None, requested, Some(requested)),
        RejectOutputDecision::Replace,
        "first frame allocates"
    );
    assert_eq!(
        reject_output_decision(Some(requested), requested, Some(requested)),
        RejectOutputDecision::Keep
    );
    let mut resized = requested;
    resized.size = UVec2::new(128, 64);
    assert_eq!(
        reject_output_decision(Some(requested), resized, Some(resized)),
        RejectOutputDecision::Replace
    );
    let mut generation = requested;
    generation.generation = 4;
    assert_eq!(
        reject_output_decision(Some(requested), generation, Some(generation)),
        RejectOutputDecision::Replace
    );
    assert_eq!(
        reject_output_decision(Some(requested), requested, None),
        RejectOutputDecision::Remove
    );
    assert_eq!(
        reject_output_decision(Some(requested), requested, Some(generation)),
        RejectOutputDecision::Remove,
        "a different depth allocation cannot reuse the mask"
    );
}

#[test]
fn default_request_map_is_empty_and_zero_rects_are_a_valid_clear_only_plan() {
    let requests = ModelRejectRequests::default();
    assert!(requests.by_render.is_empty());
    let mut view = ViewModelRejects::new(key());
    assert!(!view.full_view);
    assert!(view.rects.is_empty());
    assert!(push_projected(&mut view, ProjectedReject::Outside));
    assert!(!view.full_view);
    assert!(view.rects.is_empty());
}

#[test]
fn first_mask_publication_is_current_and_old_epochs_are_stale() {
    let masks = TemporalFramePair::new(3, [10, 20]);
    assert!(masks.current(7, 3).is_none());
    assert!(masks.previous(7, 3).is_none());
    assert!(masks.encode_current(7, 3, |slot| {
        assert_eq!(*slot, 20);
        true
    }));
    assert_eq!(masks.current(7, 3), Some(&20));
    assert!(masks.previous(7, 3).is_none());
    assert!(masks.current(8, 3).is_none());
    assert!(masks.current(7, 4).is_none());
}

#[test]
fn rejection_pair_retains_prior_and_rejects_skips_or_replacement() {
    let masks = TemporalFramePair::new(3, [10, 20]);
    assert!(masks.encode_current(1, 3, |_| true));
    assert!(masks.encode_current(2, 3, |slot| {
        assert_eq!(*slot, 10);
        true
    }));
    assert_eq!(masks.current(2, 3), Some(&10));
    assert_eq!(masks.previous(2, 3), Some(&20));

    assert!(masks.encode_current(4, 3, |_| true));
    assert!(masks.previous(4, 3).is_none());

    let replacement = TemporalFramePair::new(4, [10, 20]);
    assert!(replacement.current(1, 3).is_none());
    assert!(replacement.previous(1, 3).is_none());
    assert!(replacement.current(1, 4).is_none());
}

#[test]
fn padded_rect_contains_independently_projected_corners() {
    let origin = Vec3::ZERO;
    let b = bounds();
    let extent = UVec2::new(200, 100);
    let projected = project_reject_bounds(origin, b, Mat4::IDENTITY, clip(), extent);
    let ProjectedReject::Rect(rect) = projected else {
        panic!("expected rect");
    };
    let clip_d = clip().as_dmat4();
    for x in [b.min.x as f64, b.max.x as f64] {
        for y in [b.min.y as f64, b.max.y as f64] {
            for z in [b.min.z as f64, b.max.z as f64] {
                let c = clip_d * DVec4::new(x, y, z, 1.0);
                let uv = DVec2::new(0.5 * (c.x / c.w) + 0.5, 0.5 - 0.5 * (c.y / c.w));
                let px = uv.x * extent.x as f64;
                let py = uv.y * extent.y as f64;
                assert!(px >= rect.min.x as f64, "{px} < {}", rect.min.x);
                assert!(px <= rect.max.x as f64, "{px} > {}", rect.max.x);
                assert!(py >= rect.min.y as f64, "{py} < {}", rect.min.y);
                assert!(py <= rect.max.y as f64, "{py} > {}", rect.max.y);
            }
        }
    }
}

#[test]
fn view_state_reset_and_full_view_clear_rects() {
    let mut view = ViewModelRejects::new(key());
    view.rects.push(URect {
        min: UVec2::ZERO,
        max: UVec2::new(1, 1),
    });
    view.reject_full();
    assert!(view.full_view);
    assert!(view.rects.is_empty());
    view.reset(key());
    assert!(!view.full_view);
    assert!(view.rects.is_empty());
    assert_eq!(view.key, key());
}

fn wind_frame(gust: f32, sway: f32, tree: f32, direction: [f32; 2]) -> MonkeyFrame {
    MonkeyFrame {
        wind_dir: direction,
        wind_base_heading: 0.4,
        wind_travel: 12.0,
        wind_gust: gust,
        sway_strength: sway,
        tree_strength: tree,
        ..Default::default()
    }
}

fn wind_region(origin: Vec3, min: Vec3, max: Vec3) -> TemporalWindRegion {
    TemporalWindRegion {
        origin,
        bounds: Aabb::from_min_max(min, max),
    }
}

#[test]
fn retained_wind_projects_with_two_pixel_padding() {
    let region = wind_region(
        Vec3::ZERO,
        Vec3::new(-1.0, -1.0, -6.0),
        Vec3::new(1.0, 1.0, -4.0),
    );
    let frame = wind_frame(0.4, 1.0, 1.0, [1.0, 0.0]);
    let projected = project_retained_wind(
        region,
        Some(&frame),
        Mat4::IDENTITY,
        clip(),
        UVec2::new(200, 100),
    );
    assert!(matches!(projected, ProjectedReject::Rect(_)));
}

#[test]
fn retained_wind_envelope_contains_extreme_wave_points() {
    let frame = wind_frame(1.0, 2.0, 1.5, [1.0, 0.0]);
    let region = wind_region(
        Vec3::ZERO,
        Vec3::new(-0.1, -0.1, -10.1),
        Vec3::new(0.1, 0.1, -9.9),
    );
    let projected = project_retained_wind(
        region,
        Some(&frame),
        Mat4::IDENTITY,
        clip(),
        UVec2::new(512, 512),
    );
    let ProjectedReject::Rect(envelope) = projected else {
        panic!("expected finite envelope, got {projected:?}");
    };
    let amplitude = 0.39
        * 1.3
        * (0.75 + 0.5 * frame.wind_gust as f64).abs()
        * 1.05
        * frame.sway_strength.abs() as f64
        * frame.tree_strength.abs() as f64;
    let vertical = 0.15 * amplitude * amplitude;
    assert!(amplitude > 1.0);
    for sx in [-1.0, 1.0] {
        for sy in [-1.0, 1.0] {
            for sz in [-1.0, 1.0] {
                let point = Vec3::new(
                    (sx * amplitude) as f32,
                    (sy * vertical) as f32,
                    -10.0 + (sz * amplitude) as f32,
                );
                let point_projection = project_reject_bounds(
                    Vec3::ZERO,
                    Bounds3 {
                        min: point,
                        max: point,
                    },
                    Mat4::IDENTITY,
                    clip(),
                    UVec2::new(512, 512),
                );
                let ProjectedReject::Rect(point_rect) = point_projection else {
                    continue;
                };
                assert!(point_rect.min.x >= envelope.min.x);
                assert!(point_rect.min.y >= envelope.min.y);
                assert!(point_rect.max.x <= envelope.max.x);
                assert!(point_rect.max.y <= envelope.max.y);
            }
        }
    }
}

#[test]
fn retained_wind_different_headings_keep_finite_rectangles() {
    let region = wind_region(
        Vec3::ZERO,
        Vec3::new(-1.0, -1.0, -6.0),
        Vec3::new(1.0, 1.0, -4.0),
    );
    for direction in [[1.0, 0.0], [0.6, 0.8], [-0.8, 0.6]] {
        let frame = wind_frame(1.0, 2.0, 1.5, direction);
        let projected = project_retained_wind(
            region,
            Some(&frame),
            Mat4::IDENTITY,
            clip(),
            UVec2::new(320, 200),
        );
        assert!(
            matches!(projected, ProjectedReject::Rect(_)),
            "heading {direction:?} produced {projected:?}"
        );
    }
}

#[test]
fn retained_wind_rejects_missing_invalid_and_degenerate_inputs() {
    let region = wind_region(
        Vec3::ZERO,
        Vec3::new(-1.0, -1.0, -6.0),
        Vec3::new(1.0, 1.0, -4.0),
    );
    let good = wind_frame(0.4, 1.0, 1.0, [1.0, 0.0]);
    assert_eq!(
        project_retained_wind(region, None, Mat4::IDENTITY, clip(), UVec2::new(200, 100)),
        ProjectedReject::FullView
    );
    let mut bad = good;
    bad.wind_dir = [f32::NAN, 0.0];
    assert_eq!(
        project_retained_wind(
            region,
            Some(&bad),
            Mat4::IDENTITY,
            clip(),
            UVec2::new(200, 100)
        ),
        ProjectedReject::FullView
    );
    bad = good;
    bad.wind_dir = [-1.0e-6, 0.0];
    assert_eq!(
        project_retained_wind(
            region,
            Some(&bad),
            Mat4::IDENTITY,
            clip(),
            UVec2::new(200, 100)
        ),
        ProjectedReject::FullView
    );
    bad = good;
    bad.wind_gust = f32::MAX;
    assert_eq!(
        project_retained_wind(
            region,
            Some(&bad),
            Mat4::IDENTITY,
            clip(),
            UVec2::new(200, 100)
        ),
        ProjectedReject::FullView
    );
    let invalid_bounds = wind_region(
        Vec3::ZERO,
        Vec3::new(1.0, 0.0, 0.0),
        Vec3::new(-1.0, 0.0, 0.0),
    );
    assert_eq!(
        project_retained_wind(
            invalid_bounds,
            Some(&good),
            Mat4::IDENTITY,
            clip(),
            UVec2::new(200, 100)
        ),
        ProjectedReject::FullView
    );
    let near = wind_region(
        Vec3::ZERO,
        Vec3::new(-1.0, -1.0, -0.5),
        Vec3::new(1.0, 1.0, 0.5),
    );
    assert_eq!(
        project_retained_wind(
            near,
            Some(&good),
            Mat4::IDENTITY,
            clip(),
            UVec2::new(200, 100)
        ),
        ProjectedReject::FullView
    );
}

#[test]
fn retained_wind_large_origin_projects_and_contains_samples() {
    let origin = Vec3::new(1.0e6, 2.0e6, -1.0e6);
    let region = wind_region(
        origin,
        Vec3::new(-1.0, -1.0, -1001.0),
        Vec3::new(1.0, 1.0, -999.0),
    );
    let frame = wind_frame(1.0, 2.0, 1.5, [1.0, 0.0]);
    let extent = UVec2::new(320, 200);
    let projected = project_retained_wind(
        region,
        Some(&frame),
        Mat4::from_translation(origin),
        clip(),
        extent,
    );
    let ProjectedReject::Rect(rect) = projected else {
        panic!("large common origin must project a finite rectangle, got {projected:?}");
    };
    assert!(rect.min.x < rect.max.x && rect.min.y < rect.max.y);
    assert!(rect.max.x <= extent.x && rect.max.y <= extent.y);

    let amplitude = 0.39
        * 1.3
        * (0.75 + 0.5 * frame.wind_gust as f64).abs()
        * 1.05
        * frame.sway_strength.abs() as f64
        * frame.tree_strength.abs() as f64;
    let vertical = 0.15 * amplitude * amplitude;
    let clip_d = clip().as_dmat4();
    let samples = [
        DVec3::new(0.0, 0.0, -1000.0),
        DVec3::new(amplitude, vertical, -1000.0 + amplitude),
        DVec3::new(-amplitude, -vertical, -1000.0 - amplitude),
        DVec3::new(amplitude, -vertical, -1000.0 - amplitude),
        DVec3::new(-amplitude, vertical, -1000.0 + amplitude),
    ];
    for sample in samples {
        let clip_h = clip_d * DVec4::new(sample.x, sample.y, sample.z, 1.0);
        assert!(clip_h.w > 0.0 && clip_h.is_finite());
        let ndc = clip_h.truncate() / clip_h.w;
        let uv = DVec2::new(0.5 * ndc.x + 0.5, 0.5 - 0.5 * ndc.y);
        let px = uv.x * extent.x as f64;
        let py = uv.y * extent.y as f64;
        assert!(px >= rect.min.x as f64 && px <= rect.max.x as f64);
        assert!(py >= rect.min.y as f64 && py <= rect.max.y as f64);
    }
}
