//! Ignored host-GPU regression for the previous-rig snapshot service.
//!
//! This drives the production systems through the production `Render` schedule: a real
//! `WorldCamera` with Bevy's native-TAA mount produces demand, `copy_previous_rig_palette` submits
//! the previous-data copy, the real `upload_light`/`upload_rig_palettes` uploaders queue this
//! frame's writes, and `record_current_palette_epoch` confirms both receipts. It then reads the two
//! buffers back and checks the previous buffer still holds the prior epoch's `MonkeyFrame`, base
//! table, origin table, palette prefix and metadata while the current source has advanced.
//!
//! It needs a real GPU and is not run by the default suite:
//!
//! ```text
//! cargo test --offline --locked -p benilla-world --lib \
//!     rig_palette::temporal::snapshot::gpu_tests::gpu_previous_rig_snapshot_readback \
//!     -- --ignored --exact --nocapture
//! ```
//!
//! Nothing here is evidence that the GPU assertions pass until that command has been run on the
//! host PC.

use std::time::{Duration, Instant};

use bevy::anti_alias::taa::TemporalAntiAliasing;
use bevy::app::PluginsState;
use bevy::camera::{CameraOutputMode, PerspectiveProjection, RenderTarget};
use bevy::core_pipeline::tonemapping::Tonemapping;
use bevy::ecs::system::RunSystemOnce;
use bevy::image::Image;
use bevy::render::pipelined_rendering::PipelinedRenderingPlugin;
use bevy::render::renderer::{RenderDevice, RenderQueue};
use bevy::render::view::Hdr;
use bevy::time::TimeUpdateStrategy;
use bevy::window::{ExitCondition, WindowPlugin};
use bevy::winit::WinitPlugin;

use super::*;
use crate::aa::TemporalTarget;
use crate::lighting::SharedLightBuffer;
use crate::lighting::MONKEY_FRAME_ROWS;
use crate::rig_palette::{RigPaletteExtract, RigPalettes};
use crate::view::WorldCamera;
use benilla_assets::materials::RigTaaBinds;

const SIZE: UVec2 = UVec2::new(64, 64);

/// The two epochs' frame blocks: every row differs, so a byte read names its epoch.
const MONKEY_A: [[f32; 4]; MONKEY_FRAME_ROWS] = [[1.0, 11.0, 111.0, 1111.0]; MONKEY_FRAME_ROWS];
const MONKEY_B: [[f32; 4]; MONKEY_FRAME_ROWS] = [[2.0, 22.0, 222.0, 2222.0]; MONKEY_FRAME_ROWS];
const MONKEY_C: [[f32; 4]; MONKEY_FRAME_ROWS] = [[3.0, 33.0, 333.0, 3333.0]; MONKEY_FRAME_ROWS];

fn target_image() -> Image {
    let mut image = Image::new_target_texture(SIZE.x, SIZE.y, TextureFormat::Rgba16Float, None);
    image.texture_descriptor.usage |= TextureUsages::COPY_SRC;
    image
}

/// A headless app with the real palette plugin, the real shared-light upload registration and the
/// real render world; no window and no pipelined rendering, so `update` is one full frame.
fn app() -> App {
    let mut app = App::new();
    app.add_plugins(
        DefaultPlugins
            .set(WindowPlugin {
                primary_window: None,
                exit_condition: ExitCondition::DontExit,
                ..default()
            })
            .disable::<WinitPlugin>()
            .disable::<PipelinedRenderingPlugin>(),
    );
    app.init_resource::<crate::lighting::PropProbes>()
        .insert_resource(TimeUpdateStrategy::ManualDuration(Duration::from_secs_f64(
            1.0 / 60.0,
        )));
    crate::rig_palette::plugin(&mut app);
    crate::lighting::register_upload(&mut app);

    // Render resources exist once the render plugins have finished building.
    let deadline = Instant::now() + Duration::from_secs(30);
    while app.plugins_state() == PluginsState::Adding {
        assert!(
            Instant::now() < deadline,
            "timed out waiting for renderer readiness"
        );
        bevy::tasks::tick_global_task_pools_on_main_thread();
        std::thread::sleep(Duration::from_millis(1));
    }
    app.finish();
    app.cleanup();
    app
}

fn spawn_world_camera(app: &mut App) -> Entity {
    let target = app
        .world_mut()
        .resource_mut::<Assets<Image>>()
        .add(target_image());
    app.world_mut()
        .spawn((
            Camera3d::default(),
            Camera {
                output_mode: CameraOutputMode::Skip,
                ..default()
            },
            Projection::Perspective(PerspectiveProjection {
                fov: 60.0f32.to_radians(),
                ..default()
            }),
            RenderTarget::Image(target.into()),
            Hdr,
            Tonemapping::None,
            Msaa::Off,
            WorldCamera,
            TemporalTarget::new(SIZE, 1.0),
            // The default: `reset` is still true and no attachment exists yet.
            TemporalAntiAliasing::default(),
            Transform::from_xyz(0.0, 0.0, 5.0).looking_at(Vec3::ZERO, Vec3::Y),
        ))
        .id()
}

fn allocate(app: &mut App, bones: u32) -> (u16, u32) {
    app.world_mut()
        .resource_mut::<RigPalettes>()
        .alloc(bones)
        .expect("the palette has room")
}

fn set_origin(app: &mut App, slot: u16, origin: Vec3) {
    app.world_mut()
        .resource_mut::<RigPalettes>()
        .set_origin(slot, origin);
}

/// Write `value` into every row of `bones` bones at `base`, and mark the range dirty.
fn write_rows(app: &mut App, base: u32, bones: u32, value: f32) {
    let mut palettes = app.world_mut().resource_mut::<RigPalettes>();
    let rows = Arc::make_mut(&mut palettes.rows);
    for row in &mut rows[3 * base as usize..3 * (base + bones) as usize] {
        *row = [value; 4];
    }
    palettes.dirty.push((base, bones, false));
}

fn free(app: &mut App, slot: u16) {
    app.world_mut().resource_mut::<RigPalettes>().free(slot);
}

/// The published table, origins and live palette rows, as they should reach the GPU.
fn palette_snapshot(app: &App, watermark: u32) -> (Vec<u32>, Vec<[f32; 4]>, Vec<[f32; 4]>) {
    let palettes = app.world().resource::<RigPalettes>();
    let table = (*palettes.table).clone();
    let origins = (*palettes.origins).clone();
    let rows = palettes.rows[..3 * watermark as usize].to_vec();
    (table, origins, rows)
}

fn frame(app: &App) -> u32 {
    app.get_sub_app(RenderApp)
        .expect("render app")
        .world()
        .resource::<RigPaletteExtract>()
        .frame
        .expect("the epoch is stamped after the frame-count increment")
}

fn snapshot_state(app: &App) -> (Option<u32>, Option<u32>, bool, Option<u32>) {
    let state = app
        .get_sub_app(RenderApp)
        .expect("render app")
        .world()
        .resource::<RigTaaSnapshotState>();
    (
        state.previous.as_ref().map(|epoch| epoch.frame),
        state.copied.as_ref().map(|epoch| epoch.frame),
        state.header_valid,
        state.published.as_ref().map(|epoch| epoch.frame),
    )
}

fn u32_words(bytes: &[u8]) -> Vec<u32> {
    bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|word| u32::from_ne_bytes(*word))
        .collect()
}

fn f32_rows(bytes: &[u8]) -> Vec<[f32; 4]> {
    bytes
        .as_chunks::<16>()
        .0
        .iter()
        .map(|row| {
            [
                f32::from_ne_bytes(row[0..4].try_into().expect("4 bytes")),
                f32::from_ne_bytes(row[4..8].try_into().expect("4 bytes")),
                f32::from_ne_bytes(row[8..12].try_into().expect("4 bytes")),
                f32::from_ne_bytes(row[12..16].try_into().expect("4 bytes")),
            ]
        })
        .collect()
}

/// Copy `[offset, offset + size)` out of `source` into a mapped readback buffer.
///
/// The copy is submitted on its own: any `write_buffer` a previous frame queued is applied first, so
/// the bytes returned are the frame's final GPU state. `source` must be a legal copy source and
/// cover the range; compilation cannot check either, so the fixture checks them before the copy is
/// encoded.
fn read(
    device: &RenderDevice,
    queue: &RenderQueue,
    source: &Buffer,
    offset: u64,
    size: u64,
) -> Vec<u8> {
    assert!(
        size > 0 && size.is_multiple_of(4),
        "a mapped readback is word sized"
    );
    assert!(
        offset + size <= source.size(),
        "readback range {offset}..{} exceeds a {}-byte buffer",
        offset + size,
        source.size()
    );
    assert!(
        source.usage().contains(BufferUsages::COPY_SRC),
        "a readback source needs COPY_SRC: usage {:?}",
        source.usage()
    );
    let readback = device.create_buffer(&BufferDescriptor {
        label: Some("rig_taa_readback"),
        size,
        usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
        label: Some("rig_taa_readback"),
    });
    encoder.copy_buffer_to_buffer(source, offset, &readback, 0, size);
    let submission = queue.submit(std::iter::once(encoder.finish()));
    let deadline = Duration::from_secs(10);
    device
        .wgpu_device()
        .poll(PollType::Wait {
            submission_index: Some(submission),
            timeout: Some(deadline),
        })
        .expect("GPU wait");
    let (sender, receiver) = std::sync::mpsc::channel();
    readback.slice(..).map_async(MapMode::Read, move |result| {
        let _ = sender.send(result);
    });
    device
        .wgpu_device()
        .poll(PollType::Wait {
            submission_index: None,
            timeout: Some(deadline),
        })
        .expect("GPU map wait");
    receiver
        .recv_timeout(deadline)
        .expect("map callback")
        .expect("buffer map");
    let mut out = vec![0u8; size as usize];
    {
        let data = readback.slice(..).get_mapped_range();
        out.copy_from_slice(&data[..size as usize]);
    }
    readback.unmap();
    out
}

fn read_header(device: &RenderDevice, queue: &RenderQueue, buffer: &Buffer) -> Vec<u32> {
    u32_words(&read(device, queue, buffer, 0, HEADER_BYTES))
}

/// Write a distinguishable, nonzero header into a test-owned snapshot buffer and prove it landed:
/// a clear can only be read as a change from a state that is known not to be zero already.
fn seed_header(device: &RenderDevice, queue: &RenderQueue, buffer: &Buffer, header: [u32; 4]) {
    assert_ne!(
        header[0], 0,
        "a seeded header must be distinguishable from a clear"
    );
    queue.write_buffer(buffer, 0, bytemuck::cast_slice(&header));
    assert_eq!(
        read_header(device, queue, buffer),
        header.to_vec(),
        "the seeded header is readable before the service runs"
    );
}

fn read_slot(device: &RenderDevice, queue: &RenderQueue, buffer: &Buffer, slot: u16) -> Vec<u32> {
    u32_words(&read(
        device,
        queue,
        buffer,
        SLOT_META_OFFSET + slot as u64 * 16,
        16,
    ))
}

#[test]
#[ignore = "Requires a GPU; run explicitly on the host PC"]
fn gpu_previous_rig_snapshot_readback() {
    let mut app = app();
    let device = app.world().resource::<RenderDevice>().clone();
    let queue = app.world().resource::<RenderQueue>().clone();

    let shared = crate::lighting::new_shared_light_buffer(&device);
    let source_buffer = shared.0.clone();
    let binds = crate::rig_palette::temporal::new_rig_taa_binds(&device);
    let previous_buffer = binds.previous.clone();
    assert_ne!(source_buffer.id(), previous_buffer.id());
    app.insert_resource(shared);
    app.insert_resource(binds);

    let camera = spawn_world_camera(&mut app);

    // ---- epoch 1: the first demand frame seeds the service; nothing to copy yet.
    crate::lighting::install_test_light_data(&mut app, MONKEY_A);
    let (slot_one, base_one) = allocate(&mut app, 8);
    set_origin(&mut app, slot_one, Vec3::new(10.0, 20.0, 30.0));
    write_rows(&mut app, base_one, 8, 1.0);
    app.update();
    let first = frame(&app);
    assert_eq!(
        snapshot_state(&app),
        (Some(first), None, false, None),
        "the seed frame retains the epoch but publishes nothing"
    );
    assert_eq!(
        read_header(&device, &queue, &previous_buffer),
        [0; 4],
        "a seed frame writes no valid header"
    );
    let (table_one, origins_one, rows_one) = palette_snapshot(&app, 8);

    // ---- epoch 2: a second allocation and a write move the source; the copy must still be epoch 1.
    let (slot_two, base_two) = allocate(&mut app, 4);
    assert_ne!(slot_two, slot_one);
    assert_eq!(base_two, 8);
    set_origin(&mut app, slot_two, Vec3::new(40.0, 50.0, 60.0));
    write_rows(&mut app, base_two, 4, 2.0);
    crate::lighting::install_test_light_data(&mut app, MONKEY_B);
    app.update();
    let second = frame(&app);
    assert_eq!(second, first.wrapping_add(1));
    assert_eq!(
        snapshot_state(&app),
        (Some(second), Some(second), true, Some(second)),
        "the consecutive receipt-backed epoch publishes a valid snapshot"
    );
    assert_eq!(
        read_header(&device, &queue, &previous_buffer),
        [1, first, second, 8],
        "valid, previous frame, current frame, previous watermark"
    );

    // The previous buffer holds the prior epoch's bytes...
    let monkey = f32_rows(&read(
        &device,
        &queue,
        &previous_buffer,
        HEADER_BYTES,
        MONKEY_BYTES,
    ));
    assert_eq!(monkey, MONKEY_A.to_vec(), "previous MonkeyFrame");
    let bases = u32_words(&read(
        &device,
        &queue,
        &previous_buffer,
        SLOT_BASES_OFFSET,
        SLOT_BASES_BYTES,
    ));
    assert_eq!(bases, table_one, "previous base table");
    let origins = f32_rows(&read(
        &device,
        &queue,
        &previous_buffer,
        SLOT_ORIGINS_OFFSET,
        SLOT_ORIGINS_BYTES,
    ));
    assert_eq!(origins, origins_one, "previous origin table");
    let palette = f32_rows(&read(
        &device,
        &queue,
        &previous_buffer,
        PALETTE_OFFSET,
        8 * BONE_BYTES,
    ));
    assert_eq!(palette, rows_one, "previous live palette prefix");
    assert_eq!(
        read_slot(&device, &queue, &previous_buffer, slot_one),
        [1, 1, 0, 8],
        "the eligible slot is advertised"
    );
    assert_eq!(
        read_slot(&device, &queue, &previous_buffer, slot_two),
        [0; 4],
        "a slot allocated after the previous epoch cannot inherit validity"
    );

    // ...while the current source has advanced to epoch 2.
    let source_monkey = f32_rows(&read(
        &device,
        &queue,
        &source_buffer,
        crate::lighting::per_frame_blob_bytes() - MONKEY_BYTES,
        MONKEY_BYTES,
    ));
    assert_eq!(source_monkey, MONKEY_B.to_vec(), "current MonkeyFrame");
    let source_bases = u32_words(&read(
        &device,
        &queue,
        &source_buffer,
        crate::rig_palette::rig_table_region_offset(),
        SLOT_BASES_BYTES,
    ));
    assert_eq!(
        source_bases[slot_two as usize], base_two,
        "current base table"
    );
    assert_eq!(
        source_bases[slot_one as usize], base_one,
        "an unchanged slot keeps its base"
    );
    let source_origins = f32_rows(&read(
        &device,
        &queue,
        &source_buffer,
        crate::rig_palette::rig_origin_region_offset(),
        SLOT_ORIGINS_BYTES,
    ));
    assert_eq!(
        source_origins[slot_two as usize],
        [40.0, 50.0, 60.0, 0.0],
        "current origin table"
    );
    let (table_two, origins_two, rows_two) = palette_snapshot(&app, 12);
    let source_palette = f32_rows(&read(
        &device,
        &queue,
        &source_buffer,
        crate::rig_palette::palette_region_offset(),
        12 * BONE_BYTES,
    ));
    assert_eq!(
        &source_palette[..3 * 8],
        &rows_two[..3 * 8],
        "the untouched prefix is unchanged"
    );
    assert!(
        source_palette[3 * 8..].iter().all(|row| *row == [2.0; 4]),
        "the newly written rows carry this epoch's value"
    );
    assert_eq!(table_two[slot_two as usize], base_two);
    assert_eq!(origins_two[slot_two as usize], [40.0, 50.0, 60.0, 0.0]);

    // ---- duplicate: running both production systems again for the same epoch changes nothing.
    let header_before = read_header(&device, &queue, &previous_buffer);
    let slot_before = read_slot(&device, &queue, &previous_buffer, slot_two);
    {
        let world = app
            .get_sub_app_mut(RenderApp)
            .expect("render app")
            .world_mut();
        world
            .run_system_once(copy_previous_rig_palette)
            .expect("a duplicate copy stage");
        world
            .run_system_once(record_current_palette_epoch)
            .expect("a duplicate record stage");
    }
    assert_eq!(
        read_header(&device, &queue, &previous_buffer),
        header_before,
        "a duplicate invocation must not rotate or invalidate"
    );
    assert_eq!(
        read_slot(&device, &queue, &previous_buffer, slot_two),
        slot_before
    );
    assert_eq!(
        snapshot_state(&app),
        (Some(second), Some(second), true, Some(second))
    );

    // ---- epoch 3: the second slot is freed and re-allocated, so its identity changes.
    let reused = allocate_reusing(&mut app, slot_two, 4);
    assert_eq!(reused, slot_two, "the freed slot is reused");
    write_rows(&mut app, base_two, 4, 3.0);
    set_origin(&mut app, slot_two, Vec3::new(70.0, 80.0, 90.0));
    crate::lighting::install_test_light_data(&mut app, MONKEY_C);
    app.update();
    let third = frame(&app);
    assert_eq!(third, second.wrapping_add(1));
    assert_eq!(
        read_slot(&device, &queue, &previous_buffer, slot_two),
        [0; 4],
        "a reused slot never inherits the previous epoch's validity"
    );
    assert_eq!(
        read_slot(&device, &queue, &previous_buffer, slot_one),
        [1, 1, 0, 8],
        "an unrelated valid slot is preserved"
    );
    let previous_palette = f32_rows(&read(
        &device,
        &queue,
        &previous_buffer,
        PALETTE_OFFSET,
        12 * BONE_BYTES,
    ));
    assert!(
        previous_palette[3 * 8..].iter().all(|row| *row == [2.0; 4]),
        "the previous prefix is the pre-reuse epoch"
    );
    let source_palette = f32_rows(&read(
        &device,
        &queue,
        &source_buffer,
        crate::rig_palette::palette_region_offset(),
        12 * BONE_BYTES,
    ));
    assert!(
        source_palette[3 * 8..].iter().all(|row| *row == [3.0; 4]),
        "the current source carries the reused slot's new rows"
    );

    // ---- no demand: one invalidation, then silence.
    app.world_mut()
        .entity_mut(camera)
        .remove::<TemporalAntiAliasing>();
    app.update();
    assert_eq!(
        read_header(&device, &queue, &previous_buffer),
        [0; 4],
        "losing demand invalidates the formerly valid header"
    );
    assert_eq!(snapshot_state(&app), (None, None, false, None));
    app.update();
    assert_eq!(
        read_header(&device, &queue, &previous_buffer),
        [0; 4],
        "still off: no further submission rewrites anything"
    );

    // ---- demand returns: one seed frame, then a valid snapshot again.
    app.world_mut()
        .entity_mut(camera)
        .insert(TemporalAntiAliasing::default());
    app.update();
    let resumed = frame(&app);
    assert_eq!(snapshot_state(&app), (Some(resumed), None, false, None));
    app.update();
    let valid = frame(&app);
    assert_eq!(
        read_header(&device, &queue, &previous_buffer),
        [1, resumed, valid, 12],
        "the recovered stream publishes a consecutive snapshot"
    );

    // ---- source replacement: the new buffer has no previous contents to copy.
    let replacement = crate::lighting::new_shared_light_buffer(&device);
    let replacement_buffer = replacement.0.clone();
    assert_ne!(replacement_buffer.id(), source_buffer.id());
    app.insert_resource(replacement);
    app.update();
    assert_eq!(
        read_header(&device, &queue, &previous_buffer),
        [0; 4],
        "a replaced source cannot be labelled as the previous epoch"
    );
    let seeded = frame(&app);
    assert_eq!(
        snapshot_state(&app),
        (Some(seeded), None, false, None),
        "the replacement re-seeds"
    );
    app.update();
    let after_replacement = frame(&app);
    assert_eq!(
        read_header(&device, &queue, &previous_buffer),
        [1, seeded, after_replacement, 12],
        "a fresh source publishes again once it has one receipt-backed epoch"
    );
    let replacement_monkey = f32_rows(&read(
        &device,
        &queue,
        &replacement_buffer,
        crate::lighting::per_frame_blob_bytes() - MONKEY_BYTES,
        MONKEY_BYTES,
    ));
    assert_eq!(
        replacement_monkey,
        MONKEY_C.to_vec(),
        "the replacement source received the current frame block"
    );

    // ---- same-epoch source replacement: the frame number is unchanged, the bytes are not.
    let same_epoch = crate::lighting::new_shared_light_buffer(&device);
    install_source(&mut app, same_epoch);
    run_stages(&mut app);
    assert_eq!(
        read_header(&device, &queue, &previous_buffer),
        [0; 4],
        "a same-frame source replacement clears the old valid header"
    );
    assert_eq!(
        snapshot_state(&app),
        (None, None, false, None),
        "the replaced epoch does not keep a previous side or a publication"
    );
    // Restore the real source; the next frames seed and publish again.
    install_source(&mut app, SharedLightBuffer(replacement_buffer.clone()));
    let republished = wait_for_valid_snapshot(&mut app, &device, &queue, &previous_buffer);
    assert_eq!(republished[0], 1);

    // ---- same-epoch allocation change: the published slot records describe inputs that are gone.
    let allocations_before = {
        let world = app
            .get_sub_app_mut(RenderApp)
            .expect("render app")
            .world_mut();
        Arc::clone(&world.resource::<RigPaletteExtract>().allocations)
    };
    {
        let world = app
            .get_sub_app_mut(RenderApp)
            .expect("render app")
            .world_mut();
        let mut published = world.resource_mut::<RigPaletteExtract>();
        let mut allocations = (*published.allocations).clone();
        if let Some(slot) = allocations.iter_mut().find(|allocation| allocation.id != 0) {
            slot.id = slot.id.wrapping_add(1);
        }
        published.allocations = Arc::new(allocations);
    }
    run_stages(&mut app);
    assert_eq!(
        read_header(&device, &queue, &previous_buffer),
        [0; 4],
        "a same-frame allocation change clears the obsolete slot records"
    );
    assert!(
        !snapshot_state(&app).2,
        "the queued clear discharged the obligation"
    );
    // Put the real publication back for the following frames.
    app.get_sub_app_mut(RenderApp)
        .expect("render app")
        .world_mut()
        .resource_mut::<RigPaletteExtract>()
        .allocations = allocations_before;
    let republished = wait_for_valid_snapshot(&mut app, &device, &queue, &previous_buffer);
    assert_eq!(republished[0], 1);

    // ---- destination loss and restoration: the clear cannot be queued without a target.
    let previous_binds = app.world().resource::<RigTaaBinds>().clone();
    app.world_mut().remove_resource::<RigTaaBinds>();
    app.get_sub_app_mut(RenderApp)
        .expect("render app")
        .world_mut()
        .remove_resource::<RigTaaBinds>();
    app.update();
    assert_eq!(
        read_header(&device, &queue, &previous_buffer)[0],
        1,
        "without a destination the stale header cannot be cleared"
    );
    assert!(
        snapshot_state(&app).2,
        "the unobtainable clear is still owed, not silently retired"
    );
    // The same buffer returns: the clear is queued before anything may reuse it.
    install_destination(&mut app, previous_binds.clone());
    app.update();
    assert_eq!(
        read_header(&device, &queue, &previous_buffer),
        [0; 4],
        "restoration clears the stale validity first"
    );
    assert!(!snapshot_state(&app).2);
    let republished = wait_for_valid_snapshot(&mut app, &device, &queue, &previous_buffer);
    assert_eq!(republished[0], 1);

    // ---- unsafe invalidation targets: an invalidation is never written through them.
    //
    // wgpu's default uncaptured-error handler panics on a validation error, so reaching the end of
    // a case already says no invalid write was issued; the explicit source read below covers the
    // aliased case, where the danger is silent corruption rather than a validation error.
    // The live source is the replacement buffer the uploaders are writing; its first row is the
    // marker `install_test_light_data` writes, so a zeroed invalidation would be visible.
    let live_prefix = read(&device, &queue, &replacement_buffer, 0, HEADER_BYTES);

    // (a) The destination aliases the live source.
    install_destination(
        &mut app,
        RigTaaBinds {
            previous: replacement_buffer.clone(),
        },
    );
    run_stages(&mut app);
    assert!(
        snapshot_state(&app).2,
        "an aliased target cannot discharge the invalidation"
    );
    assert_eq!(
        read(&device, &queue, &replacement_buffer, 0, HEADER_BYTES),
        live_prefix,
        "an invalidation must never be written into the live source"
    );
    drain(&device);

    // (b) The destination is too short for even the header.
    install_destination(
        &mut app,
        RigTaaBinds {
            previous: device.create_buffer(&BufferDescriptor {
                label: Some("short rig_taa destination"),
                size: 8,
                usage: BufferUsages::COPY_DST | BufferUsages::STORAGE,
                mapped_at_creation: false,
            }),
        },
    );
    run_stages(&mut app);
    assert!(
        snapshot_state(&app).2,
        "a destination shorter than one header cannot discharge the invalidation"
    );
    drain(&device);

    // (c) The destination is not writable as a copy destination.
    install_destination(
        &mut app,
        RigTaaBinds {
            previous: device.create_buffer(&BufferDescriptor {
                label: Some("read-only rig_taa destination"),
                size: PREVIOUS_RIG_BYTES,
                usage: BufferUsages::COPY_SRC | BufferUsages::STORAGE,
                mapped_at_creation: false,
            }),
        },
    );
    run_stages(&mut app);
    assert!(
        snapshot_state(&app).2,
        "a destination without COPY_DST cannot discharge the invalidation"
    );
    drain(&device);

    // A safe destination finally takes the owed clear.
    install_destination(&mut app, previous_binds);
    run_stages(&mut app);
    assert_eq!(
        read_header(&device, &queue, &previous_buffer),
        [0; 4],
        "the owed clear lands on the first safe target"
    );
    assert!(!snapshot_state(&app).2);
    let republished = wait_for_valid_snapshot(&mut app, &device, &queue, &previous_buffer);
    assert_eq!(republished[0], 1);

    // ---- a clear written to another destination never stands in for A's (v22).
    //
    // A is valid again. B is header-writable but cannot hold a snapshot at all, and is bound while
    // demand is Off: the service's clear goes to B and is acknowledged against B.
    let a_binds = app.world().resource::<RigTaaBinds>().clone();
    assert_eq!(
        read_header(&device, &queue, &previous_buffer),
        republished,
        "A carries a valid header before the replacement"
    );
    let b = device.create_buffer(&BufferDescriptor {
        label: Some("header-only rig_taa destination"),
        size: HEADER_BYTES,
        usage: BufferUsages::COPY_SRC | BufferUsages::COPY_DST | BufferUsages::STORAGE,
        mapped_at_creation: false,
    });
    let b_buffer = b.clone();
    // B starts zero, so a zero readback would prove nothing; seed and verify a nonzero header so
    // the post-adoption readback can only mean the service actually cleared it.
    seed_header(&device, &queue, &b_buffer, [1, 21, 22, 23]);
    install_destination(&mut app, RigTaaBinds { previous: b });
    app.world_mut()
        .entity_mut(camera)
        .remove::<TemporalAntiAliasing>();
    app.update();
    assert_eq!(
        read_header(&device, &queue, &b_buffer),
        [0; 4],
        "B's own clear lands on B"
    );
    assert_eq!(snapshot_state(&app), (None, None, false, None));
    // A is untouched by all of that: its header is still the valid one.
    assert_eq!(
        read_header(&device, &queue, &previous_buffer),
        republished,
        "clearing B must not write A"
    );
    // Stable Off with B bound does no further snapshot work.
    app.update();
    assert_eq!(
        read_header(&device, &queue, &b_buffer),
        [0; 4],
        "B stays clear"
    );
    assert_eq!(snapshot_state(&app), (None, None, false, None));

    // With the source gone (no complete epoch view), binding a third destination still adopts and
    // clears it.
    app.world_mut().remove_resource::<SharedLightBuffer>();
    app.get_sub_app_mut(RenderApp)
        .expect("render app")
        .world_mut()
        .remove_resource::<SharedLightBuffer>();
    let c = crate::rig_palette::temporal::new_rig_taa_binds(&device);
    let c_buffer = c.previous.clone();
    seed_header(&device, &queue, &c_buffer, [1, 31, 32, 33]);
    install_destination(&mut app, c);
    app.update();
    assert_eq!(
        read_header(&device, &queue, &c_buffer),
        [0; 4],
        "adoption clears the new destination even without a source"
    );
    assert_eq!(snapshot_state(&app), (None, None, false, None));
    // C's clear is C's alone: A still carries its own valid header.
    assert_eq!(
        read_header(&device, &queue, &previous_buffer),
        republished,
        "clearing C must not write A either"
    );

    // Restore the source, demand and A. A's own header was never cleared by B's write, so its
    // adoption must clear it before any new valid publication is allowed.
    install_source(&mut app, SharedLightBuffer(replacement_buffer.clone()));
    install_destination(&mut app, a_binds);
    app.world_mut()
        .entity_mut(camera)
        .insert(TemporalAntiAliasing::default());
    app.update();
    assert_eq!(
        read_header(&device, &queue, &previous_buffer),
        [0; 4],
        "A's stale header is cleared on adoption, before the seed frame"
    );
    assert!(!snapshot_state(&app).2);
    // Only then does a new valid publication appear.
    let published_again = wait_for_valid_snapshot(&mut app, &device, &queue, &previous_buffer);
    assert_eq!(published_again[0], 1);
    assert_ne!(published_again[2], republished[2], "a later epoch");
}

/// Install the world's shared light buffer, as extraction would for the next stage call.
fn install_source(app: &mut App, source: SharedLightBuffer) {
    app.insert_resource(source.clone());
    app.get_sub_app_mut(RenderApp)
        .expect("render app")
        .world_mut()
        .insert_resource(source);
}

/// Install a destination binding on both worlds, as extraction would.
fn install_destination(app: &mut App, binds: RigTaaBinds) {
    app.insert_resource(binds.clone());
    app.get_sub_app_mut(RenderApp)
        .expect("render app")
        .world_mut()
        .insert_resource(binds);
}

/// Run both production snapshot systems on the render world in schedule order, without advancing
/// the frame: the same extraction a duplicate or a mid-frame input change sees.
fn run_stages(app: &mut App) {
    let world = app
        .get_sub_app_mut(RenderApp)
        .expect("render app")
        .world_mut();
    world
        .run_system_once(copy_previous_rig_palette)
        .expect("the copy stage runs");
    world
        .run_system_once(record_current_palette_epoch)
        .expect("the record stage runs");
}

/// Surface any pending GPU validation error: the default uncaptured-error handler panics, so a
/// write the service should not have issued fails the test here.
fn drain(device: &RenderDevice) {
    device
        .wgpu_device()
        .poll(PollType::Wait {
            submission_index: None,
            timeout: Some(Duration::from_secs(10)),
        })
        .expect("GPU drain");
}

/// Advance frames until the previous buffer advertises a valid snapshot again.
fn wait_for_valid_snapshot(
    app: &mut App,
    device: &RenderDevice,
    queue: &RenderQueue,
    previous: &Buffer,
) -> [u32; 4] {
    for _ in 0..8 {
        app.update();
        let header: [u32; 4] = read_header(device, queue, previous)
            .try_into()
            .expect("a four-word header");
        if header[0] == 1 {
            assert_eq!(header[2], frame(app), "the header names this frame");
            assert_eq!(header[1].wrapping_add(1), header[2]);
            assert!(
                snapshot_state(app).2,
                "a valid header is an owed-or-cleared state"
            );
            return header;
        }
    }
    panic!("the snapshot never published a valid header after restoration");
}

/// Free `slot` and allocate the same bone count again, returning the slot the allocator hands back.
fn allocate_reusing(app: &mut App, slot: u16, bones: u32) -> u16 {
    free(app, slot);
    let (reused, _) = allocate(app, bones);
    reused
}
