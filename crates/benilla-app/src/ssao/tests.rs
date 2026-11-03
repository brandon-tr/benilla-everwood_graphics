use super::*;

#[test]
fn tiers_follow_the_cvar_override_and_camera_state() {
    let mut app = App::new();
    app.init_resource::<VideoConfig>()
        .insert_resource(AoOverride(None))
        .add_systems(Update, update_ao);
    let camera = app
        .world_mut()
        .spawn((WorldCamera, Camera::default(), Camera3d::default()))
        .id();
    let preview = app
        .world_mut()
        .spawn((Camera::default(), Camera3d::default()))
        .id();
    // The registered default is Off: no pass, image unchanged.
    app.update();
    assert!(app.world().get::<AoView>(camera).is_none());
    app.world_mut()
        .resource_mut::<VideoConfig>()
        .ambient_occlusion = 1;
    app.update();
    let low = *app.world().get::<AoView>(camera).unwrap();
    assert_eq!(low.params.z, 6.0);
    assert!(app.world().get::<AoView>(preview).is_none());
    let usage = app
        .world()
        .get::<Camera3d>(camera)
        .unwrap()
        .depth_texture_usages
        .0;
    assert!(usage & TextureUsages::TEXTURE_BINDING.bits() != 0);
    app.world_mut()
        .resource_mut::<VideoConfig>()
        .ambient_occlusion = 2;
    app.update();
    let high = *app.world().get::<AoView>(camera).unwrap();
    assert_eq!(high.params.z, 12.0);
    assert!(high.params.x > low.params.x);
    app.world_mut().get_mut::<Camera>(camera).unwrap().is_active = false;
    app.update();
    assert!(app.world().get::<AoView>(camera).is_none());
    app.world_mut().get_mut::<Camera>(camera).unwrap().is_active = true;
    app.world_mut().resource_mut::<AoOverride>().0 = Some(0);
    app.update();
    assert!(app.world().get::<AoView>(camera).is_none());
}

#[test]
fn tiers_keep_the_fade_and_protection_ramps_ordered() {
    for tier in [1, 2] {
        let v = AoView::tier(tier);
        assert!(v.fade.x < v.fade.y, "distance fade runs outward");
        assert!(v.fade.z < v.fade.w, "protection ramps up with brightness");
        assert!(v.params.y > 0.0 && v.params.y < 1.0, "never a black halo");
    }
}
