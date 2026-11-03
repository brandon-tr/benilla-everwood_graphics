use super::*;
use bevy::image::TextureFormatPixelInfo as _;

/// One byte a pixel, a render attachment, never sampled.
#[test]
fn the_world_target_is_a_size_carrier_only() {
    let image = new_world_target(UVec2::new(320, 200));
    assert_eq!(image.texture_descriptor.format, TextureFormat::R8Unorm);
    let usage = image.texture_descriptor.usage;
    assert!(usage.contains(TextureUsages::RENDER_ATTACHMENT));
    assert!(
        !usage.contains(TextureUsages::TEXTURE_BINDING),
        "nothing samples the world's target: the UI camera's ground pass reads the view's \
             main texture, not this image"
    );
    assert_eq!(
        image.asset_usage,
        RenderAssetUsages::RENDER_WORLD,
        "the CPU copy of a texture nothing reads is dropped once the render world has it"
    );
}

#[test]
fn the_ui_camera_claims_the_drawing_world_camera() {
    let mut app = App::new();
    app.add_plugins(MinimalPlugins)
        .add_systems(Update, claim_backdrop);
    let ui = app
        .world_mut()
        .spawn((PlayerUiCamera, FfxBackdrop::default()))
        .id();
    let world_cam = app
        .world_mut()
        .spawn((
            WorldCamera,
            Camera {
                is_active: true,
                ..default()
            },
        ))
        .id();
    let source = |app: &mut App| app.world().get::<FfxBackdrop>(ui).unwrap().source;
    app.update();
    assert_eq!(source(&mut app), Some(world_cam), "active: claimed");
    app.world_mut()
        .get_mut::<Camera>(world_cam)
        .unwrap()
        .is_active = false;
    app.update();
    assert_eq!(source(&mut app), None, "gated: the claim goes with it");
}

/// The window's factor, not `1.0`, re-stamped when the window moves to another display.
#[test]
fn the_world_cameras_target_carries_the_windows_scale_factor() {
    let mut app = App::new();
    app.add_plugins((MinimalPlugins, bevy::asset::AssetPlugin::default()))
        .init_asset::<Image>();
    let window = |sf: f32| {
        let mut w = Window::default();
        w.resolution.set_scale_factor(sf);
        w
    };
    app.world_mut().spawn((window(2.0), PrimaryWindow));
    app.init_resource::<RenderScale>()
        .add_systems(Startup, setup_backdrop)
        // Chained as the plugin chains them: the stamp reads the size the resize pass settled.
        .add_systems(Update, (track_render_size, retarget_world_camera).chain());
    let cam = app
        .world_mut()
        .spawn((WorldCamera, RenderTarget::default()))
        .id();
    app.update();

    let stamped = |app: &App, e: Entity| match app.world().entity(e).get::<RenderTarget>() {
        Some(RenderTarget::Image(t)) => t.scale_factor,
        _ => panic!("the world camera must target the backdrop image"),
    };
    assert_eq!(
        stamped(&app, cam),
        2.0,
        "a 2× display: the target must report the window's scale factor, or every logical \
             cursor position is read as a physical one and the pick ray misses by 2×"
    );
    let temporal = *app
        .world()
        .entity(cam)
        .get::<TemporalTarget>()
        .expect("the camera's temporal target is stamped with its render target");
    assert_eq!(
        temporal.size(),
        app.world().resource::<WorldBackdrop>().render_size(),
        "the history is tied to the rendered physical size"
    );
    assert_eq!(temporal.scale(), 2.0);
    // …and it follows the window across displays.
    let mut w = app.world_mut().query::<&mut Window>();
    w.single_mut(app.world_mut())
        .unwrap()
        .resolution
        .set_scale_factor(1.0);
    app.update();
    assert_eq!(
        stamped(&app, cam),
        1.0,
        "moving to a 1× display re-stamps: a stale factor is the same defect, inverted"
    );
    assert_eq!(
        app.world()
            .entity(cam)
            .get::<TemporalTarget>()
            .unwrap()
            .scale(),
        1.0,
        "a DPI move is a target change and must invalidate temporal history"
    );
}

/// A zero-size window (minimised on some platforms) still builds a legal texture.
#[test]
fn a_degenerate_size_still_builds_a_legal_texture() {
    let image = new_world_target(UVec2::ZERO);
    assert_eq!(image.texture_descriptor.size.width, 1);
    assert_eq!(image.texture_descriptor.size.height, 1);
    assert_eq!(
        image.data.as_ref().map(Vec::len),
        TextureFormat::R8Unorm.pixel_size().ok()
    );
}

/// Scale 1.0 gives the window's own numbers exactly, which every visual golden rests on.
#[test]
fn scale_one_reproduces_the_windows_own_numbers_exactly() {
    for px in [
        UVec2::new(1280, 720),
        UVec2::new(3200, 1800),
        UVec2::new(1601, 901), // odd on both axes, the case a `/ 2` would round
    ] {
        for sf in [1.0, 1.5, 2.0] {
            assert_eq!(render_target_for(px, sf, 1.0), (px, sf), "{px} at {sf}×");
        }
    }
}

/// `logical = image_size / scale_factor` (`bevy_camera`'s `to_logical`) stays the window's at
/// every scale: x exact, y within one `round()`.
#[test]
fn every_scale_keeps_the_logical_viewport_the_windows_own() {
    for px in [UVec2::new(1280, 720), UVec2::new(3200, 1800)] {
        for sf in [1.0, 2.0] {
            let want = px.as_vec2() / sf;
            for scale in [0.25, 0.5, 0.6667, 0.75, 1.0, 1.25, 2.0, 4.0] {
                let (size, factor) = render_target_for(px, sf, scale);
                let logical = size.as_vec2() / factor;
                assert!(
                    (logical.x - want.x).abs() < 0.001,
                    "{px} at {sf}× scaled {scale}: logical width {logical:?}, want {want:?}"
                );
                assert!(
                    (logical.y - want.y).abs() < 1.0,
                    "{px} at {sf}× scaled {scale}: logical height {logical:?}, want {want:?}"
                );
            }
        }
    }
}

/// The ceiling clamps the ratio, keeping the aspect, and binds on a huge window at 1.0 too.
#[test]
fn the_axis_ceiling_shrinks_the_picture_it_does_not_reshape_it() {
    let px = UVec2::new(3840, 2160);
    let (size, _) = render_target_for(px, 1.0, 4.0);
    assert_eq!(size.x, MAX_RENDER_AXIS);
    let aspect = |v: UVec2| v.x as f32 / v.y as f32;
    assert!(
        (aspect(size) - aspect(px)).abs() < 0.001,
        "aspect kept: {size}"
    );
    // …and the clamp is not render scale's alone.
    let huge = UVec2::new(10240, 4320);
    let (size, _) = render_target_for(huge, 1.0, 1.0);
    assert!(
        size.x <= MAX_RENDER_AXIS && size.y <= MAX_RENDER_AXIS,
        "{size}"
    );
    assert!(
        (aspect(size) - aspect(huge)).abs() < 0.001,
        "aspect kept: {size}"
    );
}

#[test]
fn the_mip_bias_is_log2_below_one_and_nothing_above_it() {
    assert!((mip_bias(0.5) - -1.0).abs() < 1e-6);
    assert!((mip_bias(0.25) - -2.0).abs() < 1e-6);
    assert_eq!(
        mip_bias(1.0),
        0.0,
        "off must be exactly off — 0.0, not -0.0's cousin"
    );
    assert_eq!(mip_bias(2.0), 0.0);
    assert_eq!(mip_bias(4.0), 0.0);
}

/// The id moves, the old asset is removed, and the camera is re-stamped onto the new image.
#[test]
fn a_rebuilt_backdrop_is_a_new_asset_and_the_camera_follows_it() {
    let mut app = App::new();
    app.add_plugins((MinimalPlugins, bevy::asset::AssetPlugin::default()))
        .init_asset::<Image>();
    let mut w = Window::default();
    w.resolution.set_scale_factor(2.0);
    app.world_mut().spawn((w, PrimaryWindow));
    app.insert_resource(RenderScale(1.0))
        .add_systems(Startup, setup_backdrop)
        .add_systems(Update, (track_render_size, retarget_world_camera).chain());
    let cam = app
        .world_mut()
        .spawn((WorldCamera, RenderTarget::default()))
        .id();
    app.update();
    let first = app.world().resource::<WorldBackdrop>().image.id();

    // Move the render scale while the world is up.
    app.insert_resource(RenderScale(0.5));
    app.update();

    let second = app.world().resource::<WorldBackdrop>().image.id();
    assert_ne!(
        first, second,
        "a rebuilt backdrop is a NEW GPU texture, so it must be a new AssetId — \
             the same id hands `ui_pass` back a bind group pointing at the retired texture"
    );
    assert!(
        !app.world().resource::<Assets<Image>>().contains(first),
        "the retired image must be removed, not merely dropped: a cached material still \
             holds a strong handle to it"
    );
    match app.world().entity(cam).get::<RenderTarget>() {
        Some(RenderTarget::Image(t)) => assert_eq!(
            t.handle.id(),
            second,
            "the camera must be re-stamped onto the new image"
        ),
        _ => panic!("the world camera must target the backdrop image"),
    }
}

/// A pure window resize moves the image without moving the scale factor.
#[test]
fn a_window_resize_restamps_the_camera_even_though_the_factor_does_not_move() {
    let mut app = App::new();
    app.add_plugins((MinimalPlugins, bevy::asset::AssetPlugin::default()))
        .init_asset::<Image>();
    let mut w = Window::default();
    w.resolution.set_scale_factor(2.0);
    let win = app.world_mut().spawn((w, PrimaryWindow)).id();
    app.insert_resource(RenderScale(1.0))
        .add_systems(Startup, setup_backdrop)
        .add_systems(Update, (track_render_size, retarget_world_camera).chain());
    let cam = app
        .world_mut()
        .spawn((WorldCamera, RenderTarget::default()))
        .id();
    app.update();
    let before = match app.world().entity(cam).get::<RenderTarget>() {
        Some(RenderTarget::Image(t)) => (t.handle.id(), t.scale_factor),
        _ => panic!("the world camera must target the backdrop image"),
    };

    app.world_mut()
        .entity_mut(win)
        .get_mut::<Window>()
        .expect("the primary window")
        .resolution
        .set(640.0, 400.0);
    app.update();

    let after = match app.world().entity(cam).get::<RenderTarget>() {
        Some(RenderTarget::Image(t)) => (t.handle.id(), t.scale_factor),
        _ => panic!("the world camera must target the backdrop image"),
    };
    assert_ne!(before.0, after.0, "the resize must re-point the camera");
    assert!(
        (before.1 - after.1).abs() < 1e-6,
        "at scale 1.0 the factor is the window's own and does not move on a resize — \
             which is exactly why the re-stamp cannot be gated on it alone: {} vs {}",
        before.1,
        after.1
    );
    assert!(
        app.world().resource::<Assets<Image>>().contains(after.0),
        "the camera must point at an image that still exists"
    );
}

/// A half-scale world on a 2× display: a half-size image, a factor of 1.0, `MipBias(-1)`.
#[test]
fn a_half_scale_world_halves_the_image_the_factor_and_the_mip() {
    let mut app = App::new();
    app.add_plugins((MinimalPlugins, bevy::asset::AssetPlugin::default()))
        .init_asset::<Image>();
    let mut w = Window::default();
    w.resolution.set_scale_factor(2.0);
    let px = window_physical_size(&w);
    app.world_mut().spawn((w, PrimaryWindow));
    app.insert_resource(RenderScale(0.5))
        .add_systems(Startup, setup_backdrop)
        .add_systems(Update, (track_render_size, retarget_world_camera).chain());
    let cam = app
        .world_mut()
        .spawn((WorldCamera, RenderTarget::default()))
        .id();
    app.update();

    let backdrop = app.world().resource::<WorldBackdrop>();
    assert_eq!(backdrop.size, UVec2::new(px.x / 2, px.y / 2));
    match app.world().entity(cam).get::<RenderTarget>() {
        Some(RenderTarget::Image(t)) => assert!(
            (t.scale_factor - 1.0).abs() < 1e-6,
            "half the pixels at half the factor is the same logical viewport: {}",
            t.scale_factor
        ),
        _ => panic!("the world camera must target the backdrop image"),
    }
    let bias = app.world().entity(cam).get::<MipBias>().expect("a bias");
    assert!(
        (bias.0 - -1.0).abs() < 1e-6,
        "half scale owes one mip level: {}",
        bias.0
    );
}
