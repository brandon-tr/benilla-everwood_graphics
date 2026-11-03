use super::*;
use crate::aa::resolve::{self, TemporalResolveView};
use crate::aa::{depth, motion, reactive, rejection, scene_color};
use bevy::color::LinearRgba;
use bevy::render::camera::TemporalJitter;
use bevy::render::render_resource::Texture;

fn first_motion_texture(fixture: &mut Fixture) -> Texture {
    let main_view = fixture.camera;
    wait_stage(fixture, "first motion publication", |fixture| {
        let slot = fixture.slot(main_view)?;
        motion::gpu_probe(
            fixture.render_world(),
            slot.render,
            slot.epoch,
            slot.key.generation,
        )
        .and_then(|probe| probe.current)
    })
}

fn scene_current(fixture: &Fixture, slot: &ViewSlot) -> Option<Texture> {
    scene_color::gpu_probe(
        fixture.render_world(),
        slot.render,
        slot.epoch,
        slot.key.generation,
    )
    .and_then(|probe| probe.current)
}

fn wait_scene_current(fixture: &mut Fixture) -> (ViewSlot, Texture) {
    let main_view = fixture.camera;
    wait_stage(fixture, "scene-color current capture", |fixture| {
        let slot = fixture.slot(main_view)?;
        let texture = scene_current(fixture, &slot)?;
        Some((slot, texture))
    })
}

fn motion_pixel(fixture: &Fixture, texture: &Texture, size: UVec2) -> [f32; 4] {
    let pixels = fixture.read_texture(texture);
    pixel(&pixels, size.x / 2, size.y / 2, size)
}

fn mask_pixel(fixture: &Fixture, texture: &Texture, size: UVec2, x: u32, y: u32) -> f32 {
    let pixels = fixture.read_texture(texture);
    pixel(&pixels, x, y, size)[0]
}

fn expected_linear_rgb() -> [f32; 3] {
    let linear = LinearRgba::from(Color::srgb(0.2, 0.6, 0.9));
    [linear.red, linear.green, linear.blue]
}

fn pipelines_ready(fixture: &Fixture) -> bool {
    let world = fixture.render_world();
    depth::gpu_pipeline_ready(world)
        && motion::gpu_pipeline_ready(world)
        && reactive::gpu_pipeline_ready(world)
        && rejection::gpu_pipeline_ready(world)
        && scene_color::gpu_pipeline_ready(world)
}

fn warmup_prerequisites(fixture: &mut Fixture) {
    let main_view = fixture.camera;
    let expected = expected_linear_rgb();
    wait_stage(fixture, "pre-A1 prerequisites", |fixture| {
        let sampling = fixture.sampling_ready();
        let pipelines = pipelines_ready(fixture);
        if !sampling || !pipelines {
            fixture.warmup_evidence = format!(
                "sampling={sampling} pipelines={pipelines} pixel_checks=deferred (pipelines not ready)"
            );
            return None;
        }
        let Some(slot) = fixture.slot(main_view) else {
            fixture.warmup_evidence = format!(
                "sampling={sampling} pipelines={pipelines} depth=no-request scene=no-request"
            );
            return None;
        };
        let depth_current = depth::gpu_probe(
            fixture.render_world(),
            slot.render,
            slot.epoch,
            slot.key.generation,
        )
        .and_then(|probe| probe.current);
        let scene_current = scene_color::gpu_probe(
            fixture.render_world(),
            slot.render,
            slot.epoch,
            slot.key.generation,
        )
        .and_then(|probe| probe.current);

        let depth_center = depth_current.as_ref().map_or(f32::NAN, |texture| {
            let pixels = fixture.read_texture(texture);
            pixel(&pixels, SIZE.x / 2, SIZE.y / 2, SIZE)[0]
        });
        let source_center = if fixture
            .render_world()
            .get::<bevy::render::view::ViewTarget>(slot.render)
            .is_some()
        {
            fixture
                .read_view_source(slot.render)
                .get(((SIZE.y / 2) * SIZE.x + SIZE.x / 2) as usize)
                .copied()
        } else {
            None
        };
        let scene_center = scene_current.as_ref().map(|texture| {
            let pixels = fixture.read_texture(texture);
            pixel(&pixels, SIZE.x / 2, SIZE.y / 2, SIZE)
        });
        let depth_ok = depth_center.is_finite() && depth_center > 0.0 && depth_center <= 1.0;
        let color_ok = source_center.is_some_and(|rgba| {
            (0..3).all(|channel| is_approx(rgba[channel], expected[channel], 8.0e-3))
        });
        fixture.warmup_evidence = format!(
            "sampling={sampling} pipelines={pipelines} depth_current={} scene_current={} depth_center={depth_center:?} scene_center={scene_center:?} source_center={source_center:?} expected_linear={expected:?} depth_ok={depth_ok} color_ok={color_ok}",
            depth_current.is_some(),
            scene_current.is_some(),
        );
        (sampling
            && pipelines
            && depth_current.is_some()
            && scene_current.is_some()
            && depth_ok
            && color_ok)
            .then_some(())
    });
}

#[test]
#[ignore = "Requires a GPU; run explicitly on the host PC"]
fn gpu_temporal_pipeline_readback() {
    let mut fixture = Fixture::new(SIZE);
    fixture.print_adapter();

    // A. First publication and stationary surface.
    // Warm the actual producer pipelines and rendered subject before testing history, then
    // observe the first motion publication of the new CameraCut generation.
    warmup_prerequisites(&mut fixture);
    fixture.invalidate(AaInvalidation::CameraCut);
    let first_motion = first_motion_texture(&mut fixture);
    let first = motion_pixel(&fixture, &first_motion, SIZE);
    assert!(first[0].abs() < 1.0e-3 && first[1].abs() < 1.0e-3);
    assert!(
        first[3] < 0.5,
        "the first actual motion publication must not claim usable history: {first:?}"
    );
    println!("PASS A1: first motion publication has zero candidates and no usable history");

    let camera = fixture.camera;
    let (slot, probes) = wait_probes(&mut fixture, camera);
    assert_eq!(probes.depth.key, slot.key);
    assert_eq!(probes.scene.key, slot.key);

    let depth_current_pixels = fixture.read_texture(probes.depth.current.as_ref().unwrap());
    let depth_previous_pixels = fixture.read_texture(probes.depth.previous.as_ref().unwrap());
    let reactive_current_pixels = fixture.read_texture(probes.reactive.current.as_ref().unwrap());
    let reactive_previous_pixels = fixture.read_texture(probes.reactive.previous.as_ref().unwrap());
    let rejection_current_pixels = fixture.read_texture(probes.rejection.current.as_ref().unwrap());
    let rejection_previous_pixels =
        fixture.read_texture(probes.rejection.previous.as_ref().unwrap());
    let source = fixture.read_view_source(slot.render);
    let scene = fixture.read_texture(probes.scene.current.as_ref().unwrap());
    let motion_pixels = fixture.read_texture(probes.motion.current.as_ref().unwrap());
    let center_motion = pixel(&motion_pixels, SIZE.x / 2, SIZE.y / 2, SIZE);
    let center_depth = pixel(&depth_current_pixels, SIZE.x / 2, SIZE.y / 2, SIZE)[0];
    let previous_depth = pixel(&depth_previous_pixels, SIZE.x / 2, SIZE.y / 2, SIZE)[0];
    let center_reactive = pixel(&reactive_current_pixels, SIZE.x / 2, SIZE.y / 2, SIZE)[0];
    let previous_reactive = pixel(&reactive_previous_pixels, SIZE.x / 2, SIZE.y / 2, SIZE)[0];
    let center_rejection = pixel(&rejection_current_pixels, SIZE.x / 2, SIZE.y / 2, SIZE)[0];
    let previous_rejection = pixel(&rejection_previous_pixels, SIZE.x / 2, SIZE.y / 2, SIZE)[0];
    let source_center = pixel(&source, SIZE.x / 2, SIZE.y / 2, SIZE);
    let scene_center = pixel(&scene, SIZE.x / 2, SIZE.y / 2, SIZE);
    println!(
        "[RP02H1-A2] view={:?} key={:?} epoch={}",
        slot.render, slot.key, slot.epoch
    );
    println!("[RP02H1-A2] pipelines: {}", fixture.pipeline_diagnostics());
    println!(
        "[RP02H1-A2] motion: {}",
        motion::gpu_frame_diagnostics(fixture.render_world(), slot.render)
    );
    println!(
        "[RP02H1-A2] depth current={center_depth} previous={previous_depth}; reactive current={center_reactive} previous={previous_reactive}; rejection current={center_rejection} previous={previous_rejection}"
    );
    println!("[RP02H1-A2] source={source_center:?} captured_scene={scene_center:?} motion={center_motion:?}");
    assert!(
        center_depth.is_finite() && center_depth > 0.0 && center_depth <= 1.0,
        "captured depth is not finite/in-range at the center: {center_depth}"
    );
    assert!(is_approx(center_motion[0], 0.0, 2.0e-3));
    assert!(is_approx(center_motion[1], 0.0, 2.0e-3));
    assert!(
        (center_motion[3] - 1.0).abs() < 2.0e-2,
        "stationary center must accept history: {center_motion:?}"
    );
    println!("PASS A2: stationary center w≈1, zero motion, finite in-range depth");

    // B. Scene-color capture and pair independence.
    // HDR with Tonemapping::None means the unlit material writes linear RGB, not its sRGB
    // Color values. Rgba16Float capture and source should agree apart from half-float rounding.
    let expected_linear = LinearRgba::from(Color::srgb(0.2, 0.6, 0.9));
    let expected_rgb = [
        expected_linear.red,
        expected_linear.green,
        expected_linear.blue,
    ];
    let source = fixture.read_view_source(slot.render);
    let scene = fixture.read_texture(probes.scene.current.as_ref().unwrap());
    for (label, x, y) in [
        ("center", SIZE.x / 2, SIZE.y / 2),
        ("off-center", SIZE.x / 3, SIZE.y / 3),
    ] {
        let scene_pixel = pixel(&scene, x, y, SIZE);
        let source_pixel = pixel(&source, x, y, SIZE);
        for channel in 0..4 {
            assert!(
                is_approx(scene_pixel[channel], source_pixel[channel], 8.0e-3),
                "scene capture/source mismatch at {label} channel {channel}: {scene_pixel:?} vs {source_pixel:?}"
            );
        }
        assert_eq!(scene_pixel[3], source_pixel[3]);
        assert!(
            source_pixel[0..3].iter().all(|value| value.is_finite()),
            "expected finite linear RGB at {label}, got {source_pixel:?}"
        );
        for (channel, name, expected) in [
            (0, "red", expected_rgb[0]),
            (1, "green", expected_rgb[1]),
            (2, "blue", expected_rgb[2]),
        ] {
            assert!(
                is_approx(source_pixel[channel], expected, 8.0e-3),
                "unexpected linear {name} at {label}: actual={} expected={} (pixel {source_pixel:?})",
                source_pixel[channel],
                expected,
            );
        }
    }
    assert_ne!(
        probes.depth.current.as_ref().unwrap().id(),
        probes.depth.previous.as_ref().unwrap().id()
    );
    assert_ne!(
        probes.scene.current.as_ref().unwrap().id(),
        probes.scene.previous.as_ref().unwrap().id()
    );
    println!(
        "PASS B: scene-color capture matches the actual ViewTarget source and pair textures differ"
    );

    // C. Transparent effect appears and disappears.
    let transparent = fixture.add_transparent();
    let main_view = fixture.camera;
    let appeared = wait_stage(&mut fixture, "transparent reactive", |fixture| {
        let slot = fixture.slot(main_view)?;
        let probes = fixture.probes(&slot)?;
        let current = probes.reactive.current.as_ref()?;
        (mask_pixel(fixture, current, SIZE, SIZE.x / 2, SIZE.y / 2) > 0.5
            && probes.motion.current.is_some())
        .then_some(probes)
    });
    let appeared_motion = fixture.read_texture(appeared.motion.current.as_ref().unwrap());
    let appeared_center = pixel(&appeared_motion, SIZE.x / 2, SIZE.y / 2, SIZE);
    assert!(
        appeared_center[3] < 0.5,
        "reactive frame must reject history"
    );
    println!("PASS C1: transparent effect produced reactive=1 and motion w=0");

    fixture.app.world_mut().despawn(transparent);
    wait_stage(&mut fixture, "transparent recovery", |fixture| {
        let slot = fixture.slot(main_view)?;
        let probes = fixture.probes(&slot)?;
        let current = probes.reactive.current.as_ref()?;
        let previous = probes.reactive.previous.as_ref()?;
        let motion = probes.motion.current.as_ref()?;
        (mask_pixel(fixture, current, SIZE, SIZE.x / 2, SIZE.y / 2) < 0.5
            && mask_pixel(fixture, previous, SIZE, SIZE.x / 2, SIZE.y / 2) > 0.5
            && motion_pixel(fixture, motion, SIZE)[3] < 0.5)
            .then_some(())
    });
    println!("PASS C2: removed effect leaves current reactive=0, previous=1, motion w=0");

    wait_stage(&mut fixture, "clean stationary recovery", |fixture| {
        let slot = fixture.slot(main_view)?;
        let probes = fixture.probes(&slot)?;
        let current = probes.reactive.current.as_ref()?;
        let previous = probes.reactive.previous.as_ref()?;
        let motion = probes.motion.current.as_ref()?;
        (mask_pixel(fixture, current, SIZE, SIZE.x / 2, SIZE.y / 2) < 0.5
            && mask_pixel(fixture, previous, SIZE, SIZE.x / 2, SIZE.y / 2) < 0.5
            && motion_pixel(fixture, motion, SIZE)[3] > 0.5)
            .then_some(())
    });
    println!("PASS C3: stationary acceptance recovered after a clean consecutive epoch");

    // D. Rejection-mask scissoring from synthetic retained metadata.
    fixture.add_wind_metadata();
    let rejected = wait_stage(&mut fixture, "rejection scissoring", |fixture| {
        let slot = fixture.slot(main_view)?;
        let probes = fixture.probes(&slot)?;
        let current = probes.rejection.current.as_ref()?;
        (mask_pixel(fixture, current, SIZE, SIZE.x / 2, SIZE.y / 2) > 0.5).then_some(probes)
    });
    let mask = rejected.rejection.current.as_ref().unwrap();
    assert!(mask_pixel(&fixture, mask, SIZE, SIZE.x / 2, SIZE.y / 2) > 0.5);
    assert_eq!(mask_pixel(&fixture, mask, SIZE, 2, 2), 0.0);
    let rejected_motion = motion_pixel(&fixture, rejected.motion.current.as_ref().unwrap(), SIZE);
    assert!(rejected_motion[3] < 0.5);
    println!("PASS D: synthetic retained metadata yields center mask=1, outside=0, motion w=0");

    // E. Invalidation, resize and Off/reactivation.
    let pre_cut = fixture
        .slot(fixture.camera)
        .expect("case E pre-cut depth request");
    let generation = pre_cut.key.generation;
    println!(
        "[RP02H1-E1] before CameraCut: view={:?} key={:?} generation={generation}",
        pre_cut.render, pre_cut.key
    );
    fixture.invalidate(AaInvalidation::CameraCut);
    let (invalidated, invalidated_scene) = wait_scene_current(&mut fixture);
    println!(
        "[RP02H1-E1] after CameraCut: view={:?} key={:?} generation={}",
        invalidated.render, invalidated.key, invalidated.key.generation
    );
    assert_ne!(invalidated.key.generation, generation);
    let invalidated_probe = scene_color::gpu_probe(
        fixture.render_world(),
        invalidated.render,
        invalidated.epoch,
        invalidated.key.generation,
    )
    .expect("scene probe");
    assert!(invalidated_probe.previous.is_none());
    assert_eq!(
        invalidated_scene.id(),
        invalidated_probe.current.unwrap().id()
    );
    assert!(
        motion_pixel(
            &fixture,
            motion::gpu_probe(
                fixture.render_world(),
                invalidated.render,
                invalidated.epoch,
                invalidated.key.generation,
            )
            .and_then(|probe| probe.current)
            .as_ref()
            .unwrap(),
            SIZE,
        )[3] < 0.5
    );
    println!("PASS E1: CameraCut invalidation starts a new generation with no previous capture");

    let old_key = invalidated.key;
    let resized = UVec2::new(48, 32);
    fixture.resize(resized);
    let (resized_slot, resized_scene) = wait_scene_current(&mut fixture);
    assert_eq!(resized_slot.key.size, resized);
    assert_ne!(resized_slot.key, old_key);
    assert_eq!(resized_scene.size().width, 48);
    assert_eq!(resized_scene.size().height, 32);
    let resized_probe = scene_color::gpu_probe(
        fixture.render_world(),
        resized_slot.render,
        resized_slot.epoch,
        resized_slot.key.generation,
    )
    .expect("scene probe");
    assert!(resized_probe.previous.is_none());
    assert_eq!(resized_scene.id(), resized_probe.current.unwrap().id());
    let resized_depth = depth::gpu_probe(
        fixture.render_world(),
        resized_slot.render,
        resized_slot.epoch,
        resized_slot.key.generation,
    )
    .and_then(|probe| probe.current)
    .expect("resized depth current");
    assert_eq!(
        (resized_depth.size().width, resized_depth.size().height),
        (48, 32)
    );
    let resized_motion = motion::gpu_probe(
        fixture.render_world(),
        resized_slot.render,
        resized_slot.epoch,
        resized_slot.key.generation,
    )
    .and_then(|probe| probe.current)
    .expect("resized motion current");
    assert!(motion_pixel(&fixture, &resized_motion, resized)[3] < 0.5);
    println!("PASS E2: resize changed key/extents and started with no previous history");

    let retired_render = resized_slot.render;
    let retired_key = resized_slot.key;
    let retired_epoch = resized_slot.epoch;
    fixture.request_off();
    wait_stage(&mut fixture, "Off retirement", |fixture| {
        fixture.slot(fixture.camera).is_none().then_some(())
    });
    let world = fixture.render_world();
    assert!(
        depth::gpu_probe(world, retired_render, retired_epoch, retired_key.generation).is_none(),
        "depth output component must retire"
    );
    assert!(
        motion::gpu_probe(world, retired_render, retired_epoch, retired_key.generation).is_none(),
        "motion output component must retire"
    );
    assert!(
        reactive::gpu_probe(world, retired_render, retired_epoch, retired_key.generation).is_none(),
        "reactive output component must retire"
    );
    assert!(
        rejection::gpu_probe(world, retired_render, retired_epoch, retired_key.generation)
            .is_none(),
        "rejection output component must retire"
    );
    assert!(
        scene_color::gpu_probe(world, retired_render, retired_epoch, retired_key.generation)
            .is_none(),
        "scene-color output component must retire"
    );
    assert!(fixture.slot(fixture.camera).is_none());
    println!("PASS E3: Off retired the request and all five producer components");

    fixture.request_taa();
    let (reactivated, reactivated_scene) = wait_scene_current(&mut fixture);
    let reactivated_probe = scene_color::gpu_probe(
        fixture.render_world(),
        reactivated.render,
        reactivated.epoch,
        reactivated.key.generation,
    )
    .expect("reactivated scene probe");
    assert!(reactivated_probe.previous.is_none());
    assert_eq!(
        reactivated_scene.id(),
        reactivated_probe.current.unwrap().id()
    );
    let reactivated_motion = motion::gpu_probe(
        fixture.render_world(),
        reactivated.render,
        reactivated.epoch,
        reactivated.key.generation,
    )
    .and_then(|probe| probe.current)
    .expect("reactivated motion current");
    assert!(motion_pixel(&fixture, &reactivated_motion, resized)[3] < 0.5);
    println!("PASS E4: Taa reactivation did not reuse retired history");

    // F. Two-view isolation.
    let (second_camera, _second_target) = fixture.add_second_camera(UVec2::new(48, 32));
    let (first_slot, second_slot, first_probes, second_probes) =
        wait_stage(&mut fixture, "two-view publication", |fixture| {
            let first_slot = fixture.slot(fixture.camera)?;
            let second_slot = fixture.slot(second_camera)?;
            let first_probes = fixture.probes(&first_slot)?;
            let second_probes = fixture.probes(&second_slot)?;
            (first_probes.scene.current.is_some()
                && first_probes.scene.previous.is_some()
                && second_probes.scene.current.is_some())
            .then_some((first_slot, second_slot, first_probes, second_probes))
        });
    assert_ne!(first_slot.render, second_slot.render);
    assert_ne!(first_slot.key, second_slot.key);
    let first_scene = first_probes.scene.current.as_ref().unwrap();
    let second_scene = second_probes.scene.current.as_ref().unwrap();
    assert_ne!(first_scene.id(), second_scene.id());
    let first_pixels = fixture.read_texture(first_scene);
    let second_pixels = fixture.read_texture(second_scene);
    let first_center = pixel(&first_pixels, resized.x / 2, resized.y / 2, resized);
    let second_center = pixel(&second_pixels, 24, 16, UVec2::new(48, 32));
    assert!(
        (first_center[0] - second_center[0]).abs() > 0.05
            || (first_center[1] - second_center[1]).abs() > 0.05,
        "views did not produce independent pixels: {first_center:?} vs {second_center:?}"
    );
    let first_key = first_slot.key;
    let first_ids = [
        first_scene.id(),
        first_probes.scene.previous.as_ref().unwrap().id(),
    ];
    let second_render = second_slot.render;
    let second_key = second_slot.key;
    let second_epoch = second_slot.epoch;
    println!("PASS F1: two views have distinct keys, identities, textures and pixels");

    fixture.app.world_mut().despawn(second_camera);
    let (first_after_slot, first_after_probes) =
        wait_stage(&mut fixture, "second-view retirement", |fixture| {
            let first_slot = fixture.slot(fixture.camera)?;
            if fixture.slot(second_camera).is_some() {
                return None;
            }
            let first_probes = fixture.probes(&first_slot)?;
            first_probes.scene.current.as_ref()?;
            first_probes.scene.previous.as_ref()?;
            Some((first_slot, first_probes))
        });
    let world = fixture.render_world();
    assert!(depth::gpu_probe(world, second_render, second_epoch, second_key.generation).is_none());
    assert!(motion::gpu_probe(world, second_render, second_epoch, second_key.generation).is_none());
    assert!(
        reactive::gpu_probe(world, second_render, second_epoch, second_key.generation).is_none()
    );
    assert!(
        rejection::gpu_probe(world, second_render, second_epoch, second_key.generation).is_none()
    );
    assert!(
        scene_color::gpu_probe(world, second_render, second_epoch, second_key.generation).is_none()
    );
    assert_eq!(first_after_slot.key, first_key);
    assert_eq!(first_after_slot.render, first_slot.render);
    let after_scene = first_after_probes.scene.current.as_ref().unwrap();
    let after_previous = first_after_probes.scene.previous.as_ref().unwrap();
    let after_ids = [after_scene.id(), after_previous.id()];
    assert_ne!(after_ids[0], after_ids[1]);
    assert!(first_ids.contains(&after_ids[0]) && first_ids.contains(&after_ids[1]));
    let after_pixels = fixture.read_texture(after_scene);
    let after_center = pixel(&after_pixels, resized.x / 2, resized.y / 2, resized);
    for channel in 0..4 {
        assert!(
            is_approx(after_center[channel], first_center[channel], 8.0e-3),
            "survivor pixels changed at channel {channel}: {after_center:?} vs {first_center:?}"
        );
    }
    println!("PASS F2: retiring camera two preserved camera one's request, allocation and pixels");
}

fn set_jitter(fixture: &mut Fixture, offset: Vec2) {
    let mut jitter = fixture
        .app
        .world_mut()
        .get_mut::<TemporalJitter>(fixture.camera)
        .expect("fixture jitter component");
    jitter.offset = offset;
}

fn mask_center(fixture: &Fixture, texture: &Texture) -> f32 {
    let pixels = fixture.read_texture(texture);
    pixel(&pixels, SIZE.x / 2, SIZE.y / 2, SIZE)[0]
}

fn same_pair(first: [TextureId; 2], second: [TextureId; 2]) -> bool {
    first == second || (first[0] == second[1] && first[1] == second[0])
}

fn marker_red_count(pixels: &[[f32; 4]]) -> usize {
    let mut count = 0;
    for y in 0..SIZE.y {
        for x in SIZE.x / 2..SIZE.x {
            if pixel(pixels, x, y, SIZE)[0] > 0.5 {
                count += 1;
            }
        }
    }
    count
}

fn marker_edge(pixels: &[[f32; 4]]) -> Option<UVec2> {
    for y in 0..SIZE.y {
        for x in SIZE.x / 2..SIZE.x {
            if pixel(pixels, x, y, SIZE)[0] <= 0.5 {
                continue;
            }
            for (dx, dy) in [(-1i32, 0i32), (1, 0), (0, -1), (0, 1)] {
                let nx = x as i32 + dx;
                let ny = y as i32 + dy;
                if nx < 0 || ny < 0 || nx >= SIZE.x as i32 || ny >= SIZE.y as i32 {
                    continue;
                }
                if pixel(pixels, nx as u32, ny as u32, SIZE)[0] <= 0.1 {
                    return Some(UVec2::new(x, y));
                }
            }
        }
    }
    None
}

fn changed_marker_pixels(first: &[[f32; 4]], second: &[[f32; 4]]) -> usize {
    let mut changed = 0;
    for y in 0..SIZE.y {
        for x in SIZE.x / 2..SIZE.x {
            let a = pixel(first, x, y, SIZE);
            let b = pixel(second, x, y, SIZE);
            let a_marker = a[0] > 0.5;
            let b_marker = b[0] > 0.5;
            let difference = (0..3)
                .map(|channel| (a[channel] - b[channel]).abs())
                .fold(0.0f32, f32::max);
            if a_marker != b_marker && difference > 0.1 {
                changed += 1;
            }
        }
    }
    changed
}

#[test]
#[ignore = "Requires a GPU; run explicitly on the host PC"]
fn gpu_temporal_jitter_readback() {
    let mut fixture = Fixture::new(SIZE);
    let marker_mesh = fixture
        .app
        .world_mut()
        .resource_mut::<Assets<Mesh>>()
        .add(Rectangle::new(0.5, 2.0));
    let marker_material = fixture
        .app
        .world_mut()
        .resource_mut::<Assets<StandardMaterial>>()
        .add(StandardMaterial {
            base_color: Color::linear_rgb(0.9, 0.1, 0.1),
            unlit: true,
            cull_mode: None,
            ..default()
        });
    fixture.app.world_mut().spawn((
        Mesh3d(marker_mesh),
        MeshMaterial3d(marker_material),
        Transform::from_xyz(1.5, 0.0, 1.0).with_rotation(Quat::from_rotation_z(0.23)),
    ));
    fixture
        .app
        .world_mut()
        .entity_mut(fixture.camera)
        .insert(TemporalJitter { offset: Vec2::ZERO });
    warmup_prerequisites(&mut fixture);

    let expected = expected_linear_rgb();
    let camera = fixture.camera;
    let (marker_edge_pixel, marker_pixels) =
        wait_stage(&mut fixture, "jitter marker visible", |fixture| {
            if !fixture.sampling_ready() {
                return None;
            }
            let slot = fixture.slot(camera)?;
            let capture = scene_color::gpu_probe(
                fixture.render_world(),
                slot.render,
                slot.epoch,
                slot.key.generation,
            )?
            .current?;
            let pixels = fixture.read_texture(&capture);
            let count = marker_red_count(&pixels);
            let center = pixel(&pixels, SIZE.x / 2, SIZE.y / 2, SIZE);
            let center_ok =
                (0..3).all(|channel| is_approx(center[channel], expected[channel], 8.0e-3));
            if count >= 4 && center_ok {
                marker_edge(&pixels).map(|edge| (edge, count))
            } else {
                None
            }
        });
    println!("[RP03B1] marker pixels={marker_pixels} edge={marker_edge_pixel:?}");

    fixture.invalidate(AaInvalidation::CameraCut);
    let first = first_motion_texture(&mut fixture);
    let first_pixel = motion_pixel(&fixture, &first, SIZE);
    assert!(
        first_pixel[3] < 0.5,
        "first motion after CameraCut must not claim history: {first_pixel:?}"
    );

    let (baseline_slot, baseline_probes) = wait_probes(&mut fixture, camera);
    let baseline_key = baseline_slot.key;
    let baseline_generation = baseline_key.generation;
    let baseline_depth_ids = [
        baseline_probes.depth.current.as_ref().unwrap().id(),
        baseline_probes.depth.previous.as_ref().unwrap().id(),
    ];
    let baseline_motion_id = baseline_probes.motion.current.as_ref().unwrap().id();

    let offsets = [
        Vec2::new(0.5, 0.5),
        Vec2::new(-0.5, 0.5),
        Vec2::new(0.5, -0.5),
        Vec2::new(-0.5, -0.5),
        Vec2::new(0.25, 0.0),
        Vec2::new(-0.25, 0.0),
        Vec2::new(0.0, 0.33),
        Vec2::new(0.0, -0.33),
    ];
    let mut marker_images = Vec::new();
    for (index, offset) in offsets.into_iter().enumerate() {
        set_jitter(&mut fixture, offset);
        fixture.update();
        let slot = fixture.slot(camera).expect("jitter depth request");
        assert_eq!(slot.key, baseline_key, "jitter {index} changed the key");
        assert_eq!(slot.key.generation, baseline_generation);
        let extracted_jitter = fixture
            .render_world()
            .get::<TemporalJitter>(slot.render)
            .expect("render-world TemporalJitter");
        assert!(
            (extracted_jitter.offset - offset).length() < 1.0e-6,
            "jitter {index} requested {offset:?}, render world has {:?}",
            extracted_jitter.offset
        );
        let probes = fixture.probes(&slot).expect("jitter publications");
        let scene_pixels = fixture.read_texture(probes.scene.current.as_ref().unwrap());
        if index < 2 {
            marker_images.push(scene_pixels);
        }
        let depth_ids = [
            probes.depth.current.as_ref().unwrap().id(),
            probes.depth.previous.as_ref().unwrap().id(),
        ];
        assert!(
            same_pair(depth_ids, baseline_depth_ids),
            "jitter {index} replaced depth allocations"
        );
        assert_eq!(
            probes.motion.current.as_ref().unwrap().id(),
            baseline_motion_id,
            "jitter {index} replaced the motion allocation"
        );

        let depth_current = fixture.read_texture(probes.depth.current.as_ref().unwrap());
        let depth_previous = fixture.read_texture(probes.depth.previous.as_ref().unwrap());
        let current_depth = pixel(&depth_current, SIZE.x / 2, SIZE.y / 2, SIZE)[0];
        let previous_depth = pixel(&depth_previous, SIZE.x / 2, SIZE.y / 2, SIZE)[0];
        assert!(current_depth.is_finite() && current_depth > 0.0 && current_depth <= 1.0);
        assert!(previous_depth.is_finite() && previous_depth > 0.0 && previous_depth <= 1.0);
        assert_eq!(
            mask_center(&fixture, probes.reactive.current.as_ref().unwrap()),
            0.0
        );
        assert_eq!(
            mask_center(&fixture, probes.reactive.previous.as_ref().unwrap()),
            0.0
        );
        assert_eq!(
            mask_center(&fixture, probes.rejection.current.as_ref().unwrap()),
            0.0
        );
        assert_eq!(
            mask_center(&fixture, probes.rejection.previous.as_ref().unwrap()),
            0.0
        );

        let motion = motion_pixel(&fixture, probes.motion.current.as_ref().unwrap(), SIZE);
        assert!(
            (motion[3] - 1.0).abs() < 2.0e-2,
            "jitter {index} did not accept a stationary interior pixel: {motion:?}"
        );
        assert!(
            motion[0].abs() < 2.0e-4 && motion[1].abs() < 2.0e-4,
            "jitter {index} left unjittered stationary motion {motion:?}"
        );
        println!("[RP03B1] offset={offset:?} motion={:?}", &motion[0..2]);
    }

    assert_eq!(marker_images.len(), 2, "first two offsets must be retained");
    // A fixed warmup boundary pixel can remain inside the marker at both offsets. Count actual
    // coverage changes anywhere in the right half instead of pinning movement to that pixel.
    let changed = changed_marker_pixels(&marker_images[0], &marker_images[1]);
    assert!(
        changed > 0,
        "no marker coverage pixel changed between the two offsets"
    );
    println!("[RP03B1] marker changed_pixels={changed}");

    let last_offset = *offsets.last().unwrap();
    set_jitter(&mut fixture, last_offset);
    fixture.invalidate(AaInvalidation::CameraCut);
    let cut_first = first_motion_texture(&mut fixture);
    let cut_first_pixel = motion_pixel(&fixture, &cut_first, SIZE);
    assert!(
        cut_first_pixel[3] < 0.5,
        "first motion after a jittered cut must be invalid: {cut_first_pixel:?}"
    );
    let cut_slot = fixture.slot(camera).expect("cut request");
    assert_ne!(cut_slot.key.generation, baseline_generation);
    let cut_depth = depth::gpu_probe(
        fixture.render_world(),
        cut_slot.render,
        cut_slot.epoch,
        cut_slot.key.generation,
    )
    .expect("cut depth probe");
    assert!(
        cut_depth.previous.is_none(),
        "first frame after the jittered cut must not expose previous depth"
    );

    fixture.update();
    let recovered_slot = fixture.slot(camera).expect("recovery request");
    let recovered = fixture.probes(&recovered_slot).expect("recovery outputs");
    let recovered_motion = motion_pixel(&fixture, recovered.motion.current.as_ref().unwrap(), SIZE);
    assert!(
        (recovered_motion[3] - 1.0).abs() < 2.0e-2,
        "stationary acceptance did not recover after the jittered cut: {recovered_motion:?}"
    );
    assert!(recovered_motion[0].abs() < 2.0e-4 && recovered_motion[1].abs() < 2.0e-4);
    println!(
        "PASS: bounded Bevy jitter preserves unjittered stationary motion through a CameraCut"
    );
}
fn add_resolve_marker(fixture: &mut Fixture) {
    fixture
        .app
        .world_mut()
        .entity_mut(fixture.camera)
        .insert(TemporalResolveView);
}

fn resolve_probe(fixture: &Fixture, slot: &ViewSlot) -> Option<resolve::ResolveProbe> {
    resolve::gpu_probe(
        fixture.render_world(),
        slot.render,
        slot.epoch,
        slot.key.generation,
    )
}

fn resolve_now(fixture: &Fixture) -> Option<(ViewSlot, resolve::ResolveProbe)> {
    let slot = fixture.slot(fixture.camera)?;
    let probe = resolve_probe(fixture, &slot)?;
    assert_eq!(probe.key, slot.key);
    probe.current.as_ref()?;
    Some((slot, probe))
}

fn wait_resolve(fixture: &mut Fixture) -> (ViewSlot, resolve::ResolveProbe) {
    let camera = fixture.camera;
    wait_stage(fixture, "resolve publication", |fixture| {
        if !resolve::gpu_pipeline_ready(fixture.render_world()) {
            return None;
        }
        let slot = fixture.slot(camera)?;
        let probe = resolve_probe(fixture, &slot)?;
        assert_eq!(probe.key, slot.key);
        probe.current.as_ref()?;
        Some((slot, probe))
    })
}

fn assert_seed_matches_raw(fixture: &Fixture, slot: &ViewSlot, probe: &resolve::ResolveProbe) {
    let scene = scene_color::gpu_probe(
        fixture.render_world(),
        slot.render,
        slot.epoch,
        slot.key.generation,
    )
    .expect("scene probe");
    let raw = fixture.read_texture(scene.current.as_ref().expect("raw current"));
    let output = fixture.read_view_source(slot.render);
    for (index, (a, b)) in raw.iter().zip(output.iter()).enumerate() {
        assert!(a.iter().all(|v| v.is_finite()) && b.iter().all(|v| v.is_finite()));
        for channel in 0..3 {
            assert!(
                is_approx(a[channel], b[channel], 8.0e-3),
                "seed mismatch at {index} channel {channel}: {a:?} vs {b:?}"
            );
        }
        assert_eq!(a[3], b[3], "seed changed scene alpha at {index}");
    }
    let history = fixture.read_texture(probe.current.as_ref().expect("history"));
    let size = slot.key.size;
    let center = pixel(&history, size.x / 2, size.y / 2, size);
    assert!(
        (center[3] - 1.0).abs() < 1.0e-3,
        "seed confidence {center:?}"
    );
}

#[test]
#[ignore = "Requires a GPU; run explicitly on the host PC"]
fn gpu_temporal_resolve_readback() {
    let mut fixture = Fixture::new(SIZE);
    let marker_mesh = fixture
        .app
        .world_mut()
        .resource_mut::<Assets<Mesh>>()
        .add(Rectangle::new(0.5, 2.0));
    let marker_material = fixture
        .app
        .world_mut()
        .resource_mut::<Assets<StandardMaterial>>()
        .add(StandardMaterial {
            base_color: Color::linear_rgb(0.9, 0.1, 0.1),
            unlit: true,
            cull_mode: None,
            ..default()
        });
    fixture.app.world_mut().spawn((
        Mesh3d(marker_mesh),
        MeshMaterial3d(marker_material),
        Transform::from_xyz(1.5, 0.0, 1.0).with_rotation(Quat::from_rotation_z(0.23)),
    ));
    fixture
        .app
        .world_mut()
        .entity_mut(fixture.camera)
        .insert(TemporalJitter { offset: Vec2::ZERO });
    warmup_prerequisites(&mut fixture);
    wait_stage(
        &mut fixture,
        "resolve pipelines without marker",
        |fixture| resolve::gpu_pipeline_ready(fixture.render_world()).then_some(()),
    );
    add_resolve_marker(&mut fixture);
    fixture.invalidate(AaInvalidation::CameraCut);
    let (seed_slot, seed_probe) = wait_resolve(&mut fixture);
    assert!(seed_probe.previous.is_none(), "first resolve must seed");
    assert_seed_matches_raw(&fixture, &seed_slot, &seed_probe);
    println!("PASS A: first resolve seeds current color and confidence 1");

    let offsets = [Vec2::new(0.5, 0.5), Vec2::new(-0.5, 0.5)];
    let expected_confidence = [11.0f32, 21.0];
    let mut epochs = Vec::new();
    let mut previous_id = None;
    let mut confidence: f32 = 0.0;
    let mut edge_difference: f32 = 0.0;
    for (index, offset) in offsets.into_iter().enumerate() {
        set_jitter(&mut fixture, offset);
        fixture.update();
        let (slot, probe) = resolve_now(&fixture).expect("same-frame resolve publication");
        if let Some(previous_epoch) = epochs.last() {
            assert_eq!(
                slot.epoch,
                previous_epoch + 1,
                "resolve epoch must advance once"
            );
            assert_eq!(
                probe.previous.as_ref().map(Texture::id),
                previous_id,
                "previous history must be the preceding frame's current"
            );
        }
        let current = probe.current.as_ref().expect("history current");
        assert_ne!(
            Some(current.id()),
            probe.previous.as_ref().map(Texture::id),
            "current and previous history must be distinct slots"
        );
        epochs.push(slot.epoch);
        previous_id = Some(current.id());
        let history = fixture.read_texture(current);
        let center = pixel(&history, SIZE.x / 2, SIZE.y / 2, SIZE);
        assert!(center.iter().all(|v| v.is_finite()));
        assert!(
            (center[3] - expected_confidence[index]).abs() < 1.0e-2,
            "stationary confidence after seed must be {}, got {}",
            expected_confidence[index],
            center[3]
        );
        confidence = center[3];
        let scene = scene_color::gpu_probe(
            fixture.render_world(),
            slot.render,
            slot.epoch,
            slot.key.generation,
        )
        .expect("scene");
        let raw = fixture.read_texture(scene.current.as_ref().unwrap());
        let output = fixture.read_view_source(slot.render);
        assert!(output.iter().flatten().all(|v| v.is_finite()));
        for y in 0..SIZE.y {
            for x in SIZE.x / 2..SIZE.x {
                let a = pixel(&raw, x, y, SIZE);
                let b = pixel(&output, x, y, SIZE);
                edge_difference = edge_difference.max(
                    (0..3)
                        .map(|channel| (a[channel] - b[channel]).abs())
                        .fold(0.0f32, f32::max),
                );
            }
        }
    }
    assert!(confidence > 1.0);
    assert!(
        edge_difference > 1.0e-3,
        "resolve did not differ at a marker edge"
    );
    println!("PASS B: temporal history confidence={confidence} edge_difference={edge_difference}");

    let transparent = fixture.add_transparent();
    let (_, _, reactive_pixels, reactive_raw, reactive_output) =
        wait_stage(&mut fixture, "resolve reactive current", |fixture| {
            let slot = fixture.slot(fixture.camera)?;
            let probes = fixture.probes(&slot)?;
            if pixel(
                &fixture.read_texture(probes.reactive.current.as_ref()?),
                SIZE.x / 2,
                SIZE.y / 2,
                SIZE,
            )[0] <= 0.5
            {
                return None;
            }
            let (_, resolve) = resolve_now(fixture)?;
            let history = fixture.read_texture(resolve.current.as_ref()?);
            let scene = scene_color::gpu_probe(
                fixture.render_world(),
                slot.render,
                slot.epoch,
                slot.key.generation,
            )?;
            let raw = fixture.read_texture(scene.current.as_ref()?);
            let output = fixture.read_view_source(slot.render);
            Some((slot, resolve, history, raw, output))
        });
    let reactive_center = pixel(&reactive_pixels, SIZE.x / 2, SIZE.y / 2, SIZE);
    assert!((reactive_center[3] - 1.0).abs() < 1.0e-3);
    for (a, b) in reactive_raw.iter().zip(reactive_output.iter()) {
        assert_eq!(a, b, "current reactive rejection must preserve raw scene");
    }
    fixture.app.world_mut().despawn(transparent);
    let (reactive_previous, reactive_previous_raw, reactive_previous_output) =
        wait_stage(&mut fixture, "resolve reactive previous", |fixture| {
            let slot = fixture.slot(fixture.camera)?;
            let probes = fixture.probes(&slot)?;
            if pixel(
                &fixture.read_texture(probes.reactive.current.as_ref()?),
                SIZE.x / 2,
                SIZE.y / 2,
                SIZE,
            )[0] > 0.5
                || pixel(
                    &fixture.read_texture(probes.reactive.previous.as_ref()?),
                    SIZE.x / 2,
                    SIZE.y / 2,
                    SIZE,
                )[0] <= 0.5
            {
                return None;
            }
            let (_, resolve) = resolve_now(fixture)?;
            let history = fixture.read_texture(resolve.current.as_ref()?);
            let scene = scene_color::gpu_probe(
                fixture.render_world(),
                slot.render,
                slot.epoch,
                slot.key.generation,
            )?;
            let raw = fixture.read_texture(scene.current.as_ref()?);
            let output = fixture.read_view_source(slot.render);
            Some((history, raw, output))
        });
    let reactive_previous_center = pixel(&reactive_previous, SIZE.x / 2, SIZE.y / 2, SIZE);
    assert!((reactive_previous_center[3] - 1.0).abs() < 1.0e-3);
    for (a, b) in reactive_previous_raw
        .iter()
        .zip(reactive_previous_output.iter())
    {
        assert_eq!(a, b, "previous reactive rejection must preserve raw scene");
    }
    fixture.update();
    let (_, clean) = resolve_now(&fixture).expect("clean direct resolve frame");
    assert!(
        pixel(
            &fixture.read_texture(clean.current.as_ref().unwrap()),
            SIZE.x / 2,
            SIZE.y / 2,
            SIZE,
        )[3] > 1.0
    );
    println!("PASS C: current/previous reactive rejection resets then recovers confidence");

    fixture.invalidate(AaInvalidation::CameraCut);
    let (cut_slot, cut_probe) = wait_resolve(&mut fixture);
    assert_ne!(cut_slot.key.generation, seed_slot.key.generation);
    assert!(cut_probe.previous.is_none());
    assert_seed_matches_raw(&fixture, &cut_slot, &cut_probe);
    let resized = UVec2::new(48, 32);
    fixture.resize(resized);
    let (resized_slot, resized_probe) = wait_resolve(&mut fixture);
    assert_eq!(resized_slot.key.size, resized);
    assert!(resized_probe.previous.is_none());
    assert_seed_matches_raw(&fixture, &resized_slot, &resized_probe);
    println!("PASS D: CameraCut and resize replace history and seed");

    // Capture the raw assembled scene while the temporal lanes still run, then leave TAA. The
    // production marker belongs to the AA plugin now: the effective-mode change retires it on the
    // next `Last`, the resolve must drop its history, and the view must fall back to that raw scene.
    let camera = fixture.camera;
    let raw_slot = fixture.slot(camera).expect("slot before retirement");
    let scene = scene_color::gpu_probe(
        fixture.render_world(),
        raw_slot.render,
        raw_slot.epoch,
        raw_slot.key.generation,
    )
    .expect("raw scene before retirement");
    let raw = fixture.read_texture(scene.current.as_ref().unwrap());

    fixture
        .app
        .world_mut()
        .resource_mut::<AaPolicy>()
        .request(AaMode::Msaa);
    wait_stage(&mut fixture, "resolve retirement", |fixture| {
        let Some(slot) = fixture.slot(fixture.camera) else {
            return Some(());
        };
        resolve_probe(fixture, &slot).is_none().then_some(())
    });
    let output = fixture.read_view_source(raw_slot.render);
    for (a, b) in raw.iter().zip(output.iter()) {
        assert_eq!(a, b, "retired resolve must leave raw output untouched");
    }
    println!("PASS E: leaving TAA retires the marker and resolve history, leaving raw output");
}
