use super::*;

#[test]
fn alloc_free_coalesce_roundtrip() {
    let mut p = RigPalettes::default();
    let (s1, b1) = p.alloc(10).unwrap();
    let (s2, b2) = p.alloc(20).unwrap();
    let (s3, b3) = p.alloc(30).unwrap();
    assert!(s1 >= 1, "slot 0 is the no-rig sentinel");
    assert_eq!((b1, b2, b3), (0, 10, 30));
    p.free(s2);
    assert_eq!(p.occupancy().0, 2);
    // An exact fit reuses the freed middle range.
    let (s4, b4) = p.alloc(20).unwrap();
    assert_eq!(b4, 10);
    p.free(s1);
    p.free(s4);
    p.free(s3);
    assert_eq!(p.free_ranges, vec![(0, MAX_PALETTE_BONES as u32)]);
    assert_eq!(p.occupancy().0, 0);
    assert_eq!(p.occupancy().2, 3, "peak slots");
    assert_eq!(p.occupancy().3, 60, "peak bones");
}

#[test]
fn freed_rows_read_zero_not_stale() {
    let mut p = RigPalettes::default();
    let (slot, base) = p.alloc(2).unwrap();
    Arc::make_mut(&mut p.rows)[3 * base as usize] = [1.0; 4];
    p.free(slot);
    assert_eq!(p.rows[3 * base as usize], [0.0; 4]);
    // The free is a dirty range too, so the GPU copy zeroes as well.
    assert!(p.dirty.contains(&(base, 2, false)));
}

/// The copy-on-write clone copies only the live prefix; a free while shared still zeroes.
#[test]
fn a_shared_arc_write_copies_only_the_live_prefix() {
    let mut p = RigPalettes::default();
    let (_s1, b1) = p.alloc(2).unwrap();
    let (s2, b2) = p.alloc(3).unwrap();
    for r in 3 * b1 as usize..3 * (b1 + 2) as usize {
        Arc::make_mut(&mut p.rows)[r] = [1.0; 4];
    }
    for r in 3 * b2 as usize..3 * (b2 + 3) as usize {
        Arc::make_mut(&mut p.rows)[r] = [2.0; 4];
    }
    assert_eq!(p.bone_watermark(), 5, "first-fit packs the content low");
    // The extract's held reference, as at every frame's first write.
    let held = p.rows.clone();
    let wm = p.bone_watermark();
    let rows = rows_make_mut(
        &mut p.rows,
        &mut p.spare,
        wm,
        &mut p.cost_copies,
        &mut p.cost_copy_us,
    );
    rows[0] = [9.0; 4];
    assert_eq!(p.cost_copies, 1, "the shared write copied");
    assert_eq!(
        p.rows[3 * b2 as usize],
        [2.0; 4],
        "the neighbour survived the prefix copy"
    );
    assert_eq!(held[0], [1.0; 4], "the held (extracted) side is untouched");
    // Free the top rig while shared: the zeroing must land inside the copied prefix.
    let held2 = p.rows.clone();
    p.free(s2);
    assert_eq!(
        p.rows[3 * b2 as usize],
        [0.0; 4],
        "freed rows zeroed through the COW"
    );
    assert_eq!(
        held2[3 * b2 as usize],
        [2.0; 4],
        "the held side keeps the old frame"
    );
    assert_eq!(
        p.bone_watermark(),
        2,
        "the tail grew back down to the live rig"
    );
    // A fresh shared write after the shrink still carries the survivor whole.
    let _held3 = p.rows.clone();
    let wm = p.bone_watermark();
    let rows = rows_make_mut(
        &mut p.rows,
        &mut p.spare,
        wm,
        &mut p.cost_copies,
        &mut p.cost_copy_us,
    );
    rows[1] = [7.0; 4];
    assert_eq!(
        p.rows[0], [9.0; 4],
        "the live rig's rows survive every bounded copy"
    );
}

#[test]
fn alloc_fails_gracefully_at_capacity() {
    let mut p = RigPalettes::default();
    let (big, _) = p.alloc(MAX_PALETTE_BONES as u32).unwrap();
    assert!(p.alloc(1).is_none(), "no bones left");
    p.free(big);
    // Slot exhaustion, separately from bone exhaustion.
    let taken: Vec<u16> = (0..MAX_RIG_SLOTS - 1)
        .map(|_| p.alloc(1).unwrap().0)
        .collect();
    assert!(p.alloc(1).is_none(), "no slots left");
    for s in taken {
        p.free(s);
    }
    assert!(p.alloc(1).is_some());
}

#[test]
fn a_rider_frame_fills_every_row_and_re_writing_it_is_free() {
    let mut p = RigPalettes::default();
    let skin = RigSkin::allocate_bones(&mut p, 4, Handle::default()).unwrap();
    let origin = Vec3::new(-9464.31, 62.17, 56.91);
    let frame =
        Affine3A::from_rotation_translation(Quat::from_rotation_y(0.7), Vec3::new(0.3, 1.4, -0.2));
    p.dirty.clear();
    p.write_rider(skin.slot, frame, origin);
    assert_eq!(p.dirty.len(), 1, "the first write dirties the range");
    assert_eq!(p.dirty[0].1, 4, "and dirties ALL of it, not just row 0");

    let pal = p.world_palette(skin.slot, 4).unwrap();
    let want = Mat4::from(Affine3A::from_translation(origin) * frame);
    for (b, m) in pal.iter().enumerate() {
        assert!(m.abs_diff_eq(want, 1e-2), "row {b}: {m:?} vs {want:?}");
    }
    // And the unsummed pair the vertex stage actually reads.
    let (o, t) = p.rider_placement(skin.slot).unwrap();
    assert_eq!(o, origin);
    assert!(t.abs_diff_eq(Vec3::from(frame.translation), 1e-6));

    p.dirty.clear();
    p.write_rider(skin.slot, frame, origin);
    assert!(
        p.dirty.is_empty(),
        "an unchanged rider frame must not dirty its range"
    );
    // A sub-epsilon move still lands: bit equality, not an epsilon.
    p.write_rider(
        skin.slot,
        Affine3A::from_translation(Vec3::new(0.3, 1.4, -0.2 + 1.0e-6)) * frame,
        origin,
    );
    assert_eq!(p.dirty.len(), 1, "a sub-millimetre move is still a move");
}

#[test]
fn world_palette_reconstructs_the_affine_including_the_rig_origin() {
    let mut p = RigPalettes::default();
    let (slot, base) = p.alloc(1).unwrap();
    // A 90° turn about Y and a (1, 2, 3) offset, in a rig far from the map origin.
    let origin = Vec3::new(-9464.31, 62.17, 56.91);
    let m = Mat4::from_rotation_translation(
        Quat::from_rotation_y(std::f32::consts::FRAC_PI_2),
        Vec3::new(1.0, 2.0, 3.0),
    );
    let a = Affine3A::from_mat4(m);
    p.set_origin(slot, origin);
    let rows = Arc::make_mut(&mut p.rows);
    let (m3, t) = (a.matrix3, a.translation);
    let r = 3 * base as usize;
    rows[r] = [m3.x_axis.x, m3.y_axis.x, m3.z_axis.x, t.x];
    rows[r + 1] = [m3.x_axis.y, m3.y_axis.y, m3.z_axis.y, t.y];
    rows[r + 2] = [m3.x_axis.z, m3.y_axis.z, m3.z_axis.z, t.z];
    let want = Mat4::from_translation(origin) * m;
    let pal = p.world_palette(slot, 1).unwrap();
    assert!(pal[0].abs_diff_eq(want, 1e-2), "{:?} vs {want:?}", pal[0]);
    let q = Vec3::new(0.5, -1.0, 2.0);
    assert!(pal[0]
        .transform_point3(q)
        .abs_diff_eq(want.transform_point3(q), 1e-2));
    // The origin is load-bearing here, not a rounding-sized detail.
    assert!(
        p.world_palette(slot, 1).unwrap()[0]
            .w_axis
            .truncate()
            .distance(origin + Vec3::new(1.0, 2.0, 3.0))
            < 1e-2
    );
}

/// `rig_row` is the non-allocating single-bone counterpart of `rig_rows`.
#[test]
fn rig_row_matches_rig_rows_and_rejects_invalid_reads() {
    let mut p = RigPalettes::default();
    let (slot, base) = p.alloc(2).unwrap();
    let mats = [
        Mat4::from_rotation_translation(Quat::from_rotation_y(0.4), Vec3::new(1.0, 2.0, 3.0)),
        Mat4::from_scale_rotation_translation(
            Vec3::new(2.0, 0.5, 1.5),
            Quat::from_rotation_x(-0.2),
            Vec3::new(-4.0, 5.0, 6.0),
        ),
    ];
    let rows = Arc::make_mut(&mut p.rows);
    for (bone, m) in mats.iter().enumerate() {
        let a = Affine3A::from_mat4(*m);
        let (m3, t) = (a.matrix3, a.translation);
        let r = 3 * (base as usize + bone);
        rows[r] = [m3.x_axis.x, m3.y_axis.x, m3.z_axis.x, t.x];
        rows[r + 1] = [m3.x_axis.y, m3.y_axis.y, m3.z_axis.y, t.y];
        rows[r + 2] = [m3.x_axis.z, m3.y_axis.z, m3.z_axis.z, t.z];
    }

    let all = p.rig_rows(slot, 2).unwrap();
    for (bone, want) in all.iter().enumerate() {
        let got = p.rig_row(slot, bone as u16).unwrap();
        assert_eq!(&got, want, "bone {bone}");
        assert_eq!(p.rig_row(slot, bone as u16).unwrap(), got, "repeat");
    }
    assert!(p.rig_row(slot, 2).is_none(), "out-of-range bone");
    assert!(p.rig_row(0, 0).is_none(), "slot 0 is the no-rig sentinel");
    assert!(p.rig_row(MAX_RIG_SLOTS as u16, 0).is_none(), "unknown slot");

    let old = p.rig_row(slot, 0).unwrap();
    p.free(slot);
    assert!(p.rig_row(slot, 0).is_none(), "freed slot");
    let (slot2, _) = p.alloc(1).unwrap();
    assert_eq!(slot2, slot, "the freed slot is reused");
    let fresh = p.rig_row(slot2, 0).unwrap();
    assert!(
        fresh.transform_point3(Vec3::new(1.0, 2.0, 3.0)).length() < 1e-6,
        "fresh allocation has no prior occupant transform"
    );
    p.write_rider(
        slot2,
        Affine3A::from_rotation_translation(Quat::from_rotation_z(0.8), Vec3::new(-7.0, 8.0, 9.0)),
        Vec3::ZERO,
    );
    assert_ne!(p.rig_row(slot2, 0).unwrap(), old);
}

/// A slot recycles: the freed rig's origin must not ride into the next unit that claims it.
#[test]
fn freeing_a_slot_clears_its_origin() {
    let mut p = RigPalettes::default();
    let (slot, _) = p.alloc(4).unwrap();
    p.set_origin(slot, Vec3::new(-9464.31, 62.17, 56.91));
    assert_ne!(p.origins[slot as usize], [0.0; 4]);
    p.free(slot);
    assert_eq!(p.origins[slot as usize], [0.0; 4]);
}

#[test]
fn dirty_ranges_coalesce_into_runs() {
    // Adjacent and overlapping merge, a hole splits, input order does not matter.
    assert_eq!(
        coalesce_ranges_with_gap(vec![(70, 5), (0, 10), (10, 20), (25, 10), (40, 5)], 0),
        vec![(0, 35), (40, 5), (70, 5)]
    );
    assert_eq!(coalesce_ranges(Vec::new()), Vec::new());
}

#[test]
fn dirty_ranges_bridge_small_gaps_but_not_large_ones() {
    // A parked rig between two live ones is bridged; a wider gap still splits.
    assert_eq!(
        coalesce_ranges_with_gap(vec![(0, 10), (14, 6), (30, 5)], 4),
        vec![(0, 20), (30, 5)]
    );
    // The shipped tolerance bridges a gap of exactly its size and no more.
    let g = COALESCE_GAP_BONES;
    assert_eq!(
        coalesce_ranges(vec![(0, 10), (10 + g, 5)]),
        vec![(0, 15 + g)]
    );
    assert_eq!(
        coalesce_ranges(vec![(0, 10), (11 + g, 5)]),
        vec![(0, 10), (11 + g, 5)]
    );
}

#[test]
fn region_layout_is_consistent() {
    // `wow_model.wgsl` mirrors this layout, and wgpu checks the bound size against its struct
    // at draw time, so a mismatch fails there.
    assert_eq!(
        crate::instance_tint::region_offset() - rig_table_region_offset(),
        (MAX_RIG_SLOTS * 4) as u64,
        "the tint table follows the slot table"
    );
    assert_eq!(
        rig_origin_region_offset() - crate::instance_tint::region_offset(),
        crate::instance_tint::region_bytes(),
        "the rig-origin table follows the tint table"
    );
    assert_eq!(
        crate::mat_anim_table::region_offset() - rig_origin_region_offset(),
        rig_origin_region_bytes(),
        "the mat-anim table follows the origin table"
    );
    assert_eq!(
        crate::straddle::region_offset() - crate::mat_anim_table::region_offset(),
        crate::mat_anim_table::region_bytes(),
        "the straddle clip table follows the mat-anim table"
    );
    assert_eq!(
        crate::instance_surface::region_offset() - crate::straddle::region_offset(), // MONKEY: new table.
        crate::straddle::region_bytes(),
        "the surface table follows the straddle clip table" // MONKEY: layout fence.
    );
    // MONKEY: vec4 surface rows directly precede the runtime palette array.
    assert_eq!(
        palette_region_offset() - crate::instance_surface::region_offset(),
        crate::instance_surface::region_bytes()
    );
    assert_eq!(
        crate::instance_surface::region_bytes(),
        (MAX_RIG_SLOTS * 16) as u64
    );
    assert_eq!(
        palette_regions_bytes(),
        (MAX_RIG_SLOTS * 4) as u64
                + crate::instance_tint::region_bytes()
                + rig_origin_region_bytes()
                + crate::mat_anim_table::region_bytes()
                + crate::straddle::region_bytes()
                + crate::instance_surface::region_bytes() // MONKEY: full buffer size.
                + (MAX_PALETTE_BONES as u64) * BONE_BYTES
    );
    // One `vec4` a slot: the shader declares `array<vec4<f32>, 2048>`.
    assert_eq!(rig_origin_region_bytes(), (MAX_RIG_SLOTS * 16) as u64);
    // One word a slot: the shader declares `array<u32, 2048>` for both tables.
    assert_eq!(
        crate::instance_tint::region_bytes(),
        (MAX_RIG_SLOTS * 4) as u64
    );
}

#[test]
fn allocation_identity_changes_when_slot_and_range_are_reused() {
    let mut p = RigPalettes::default();
    let (slot, base) = p.alloc(8).unwrap();
    let first = p.allocation_ids[slot as usize];
    assert_ne!(first, 0);
    p.free(slot);
    assert_eq!(p.allocation_ids[slot as usize], 0);
    let counter = p.next_allocation_id;
    let slots = p.free_slots.clone();
    let ranges = p.free_ranges.clone();
    let generation = p.table_generation;
    p.free(slot);
    assert_eq!(p.next_allocation_id, counter);
    assert_eq!(p.free_slots, slots);
    assert_eq!(p.free_ranges, ranges);
    assert_eq!(p.table_generation, generation);
    assert_eq!(p.allocation_ids[slot as usize], 0);
    let (slot2, base2) = p.alloc(8).unwrap();
    assert_eq!((slot2, base2), (slot, base));
    assert_ne!(p.allocation_ids[slot2 as usize], 0);
    assert_ne!(p.allocation_ids[slot2 as usize], first);
}

#[test]
fn failed_allocations_do_not_consume_identity() {
    let mut p = RigPalettes::default();
    let before = p.next_allocation_id;
    assert!(p.alloc(0).is_none());
    assert_eq!(p.next_allocation_id, before);
    assert!(p.alloc(MAX_PALETTE_BONES as u32 + 1).is_none());
    assert_eq!(p.next_allocation_id, before);
}

#[test]
fn allocation_identity_exhaustion_preserves_allocator() {
    let mut p = RigPalettes::default();
    p.alloc(4).unwrap();
    p.next_allocation_id = u64::MAX;
    let ranges = p.free_ranges.clone();
    let slots = p.free_slots.clone();
    let high = p.slot_high;
    let generation = p.table_generation;
    let live = p.live_bones;
    let ids = p.allocation_ids.clone();
    let lens = p.slot_len.clone();
    let table = (*p.table).clone();
    let rows = Arc::as_ptr(&p.rows);
    assert!(p.alloc(4).is_none());
    assert_eq!(p.free_ranges, ranges);
    assert_eq!(p.free_slots, slots);
    assert_eq!(p.slot_high, high);
    assert_eq!(p.table_generation, generation);
    assert_eq!(p.live_bones, live);
    assert_eq!(p.allocation_ids, ids);
    assert_eq!(p.slot_len, lens);
    assert_eq!(*p.table, table);
    assert_eq!(Arc::as_ptr(&p.rows), rows);
    assert_eq!(p.next_allocation_id, u64::MAX);
}

#[test]
fn published_allocations_match_the_captured_palette() {
    let mut app = App::new();
    app.insert_resource(RigPalettes::default())
        .insert_resource(RigPaletteExtract::default())
        .add_systems(PostUpdate, publish_rig_palettes);
    let (first_slot, first_id, first_base, freed_slot, freed_base) = {
        let mut p = app.world_mut().resource_mut::<RigPalettes>();
        let (first_slot, first_base) = p.alloc(8).unwrap();
        let first_id = p.allocation_ids[first_slot as usize];
        let (freed_slot, freed_base) = p.alloc(4).unwrap();
        assert_ne!(freed_base, 0);
        (first_slot, first_id, first_base, freed_slot, freed_base)
    };
    app.update();
    {
        let p = app.world().resource::<RigPalettes>();
        let out = app.world().resource::<RigPaletteExtract>();
        assert_eq!(out.allocations.len(), MAX_RIG_SLOTS);
        assert_eq!(out.allocations[first_slot as usize].id, first_id);
        assert_eq!(out.allocations[first_slot as usize].base, first_base);
        assert_eq!(out.allocations[first_slot as usize].bones, 8);
        assert_eq!(out.allocations[freed_slot as usize].bones, 4);
        assert_eq!(out.watermark, p.bone_watermark());
    }
    app.world_mut()
        .resource_mut::<RigPalettes>()
        .free(freed_slot);
    app.update();
    let p = app.world().resource::<RigPalettes>();
    let out = app.world().resource::<RigPaletteExtract>();
    assert_eq!(p.table[freed_slot as usize], freed_base);
    assert_eq!(out.table[freed_slot as usize], freed_base);
    assert_eq!(
        out.allocations[freed_slot as usize],
        RigAllocation::default()
    );
    assert_eq!(out.allocations[first_slot as usize].id, first_id);
    assert_eq!(out.allocations[first_slot as usize].base, first_base);
    assert_eq!(out.allocations[first_slot as usize].bones, 8);
    assert_eq!(out.allocations[0], RigAllocation::default());
    assert_eq!(out.watermark, p.bone_watermark());
}
#[test]
fn unchanged_palette_advances_epoch_without_rebuilding_snapshots() {
    let mut app = App::new();
    app.add_plugins(bevy::diagnostic::FrameCountPlugin)
        .insert_resource(RigPalettes::default())
        .insert_resource(RigPaletteExtract::default())
        .add_systems(PostUpdate, publish_rig_palettes)
        .add_systems(
            Last,
            stamp_rig_palette_epoch.after(bevy::diagnostic::update_frame_count),
        );
    app.update();
    let first = app.world().resource::<RigPaletteExtract>();
    let first_allocs = Arc::clone(&first.allocations);
    let first_rows = Arc::clone(&first.rows);
    let first_frame = first.frame;
    assert_eq!(
        first_frame,
        Some(app.world().resource::<bevy::diagnostic::FrameCount>().0)
    );
    app.update();
    let second = app.world().resource::<RigPaletteExtract>();
    assert_eq!(
        second.frame,
        Some(app.world().resource::<bevy::diagnostic::FrameCount>().0)
    );
    assert_eq!(second.frame, Some(first_frame.unwrap().wrapping_add(1)));
    assert!(Arc::ptr_eq(&first_allocs, &second.allocations));
    assert!(Arc::ptr_eq(&first_rows, &second.rows));
}

#[test]
fn missing_frame_count_marks_epoch_unavailable() {
    let mut app = App::new();
    let extract = RigPaletteExtract {
        frame: Some(42),
        ..Default::default()
    };
    app.insert_resource(extract)
        .add_systems(Last, stamp_rig_palette_epoch);
    app.update();
    assert_eq!(app.world().resource::<RigPaletteExtract>().frame, None);
}

#[test]
fn frame_stamp_matches_bevy_counter_wrap() {
    let mut app = App::new();
    app.add_plugins(bevy::diagnostic::FrameCountPlugin)
        .insert_resource(RigPaletteExtract::default())
        .add_systems(
            Last,
            stamp_rig_palette_epoch.after(bevy::diagnostic::update_frame_count),
        );
    app.world_mut()
        .resource_mut::<bevy::diagnostic::FrameCount>()
        .0 = u32::MAX;
    app.update();
    assert_eq!(app.world().resource::<RigPaletteExtract>().frame, Some(0));
}

#[test]
fn pose_only_publication_reuses_allocation_metadata() {
    let mut app = App::new();
    app.insert_resource(RigPalettes::default())
        .insert_resource(RigPaletteExtract::default())
        .add_systems(PostUpdate, publish_rig_palettes);
    app.world_mut()
        .resource_mut::<RigPalettes>()
        .alloc(4)
        .unwrap();
    app.update();
    let first = Arc::clone(&app.world().resource::<RigPaletteExtract>().allocations);
    {
        let mut p = app.world_mut().resource_mut::<RigPalettes>();
        Arc::make_mut(&mut p.rows)[0] = [1.0; 4];
        p.dirty.push((0, 1, false));
    }
    app.update();
    let second = app.world().resource::<RigPaletteExtract>();
    assert!(Arc::ptr_eq(&first, &second.allocations));
    assert!(second.allocations[1].id != 0);
}

fn late_allocate_once(mut done: Local<bool>, mut palettes: ResMut<RigPalettes>) {
    if !*done {
        *done = true;
        palettes.alloc(6).unwrap();
    }
}

#[test]
fn late_allocation_does_not_relabel_an_older_publication() {
    let mut app = App::new();
    app.add_plugins(bevy::diagnostic::FrameCountPlugin)
        .insert_resource(RigPalettes::default())
        .insert_resource(RigPaletteExtract::default())
        .add_systems(PostUpdate, publish_rig_palettes)
        .add_systems(
            Last,
            (
                late_allocate_once,
                stamp_rig_palette_epoch.after(bevy::diagnostic::update_frame_count),
            )
                .chain(),
        );
    let first = {
        let mut p = app.world_mut().resource_mut::<RigPalettes>();
        p.alloc(4).unwrap()
    };
    let first_id = app.world().resource::<RigPalettes>().allocation_ids[first.0 as usize];
    app.update();
    let live = {
        let p = app.world().resource::<RigPalettes>();
        (1..p.slot_high)
            .filter(|&s| p.slot_len[s] > 0)
            .map(|s| (s as u16, p.allocation_ids[s], p.table[s], p.slot_len[s]))
            .collect::<Vec<_>>()
    };
    assert_eq!(live.len(), 2);
    assert_eq!(app.world().resource::<RigPalettes>().bone_watermark(), 10);
    assert_eq!(app.world().resource::<RigPalettes>().bone_watermark(), 10);
    let late = live
        .iter()
        .copied()
        .find(|&(slot, _, _, _)| slot != first.0)
        .expect("late allocation");
    let old = app.world().resource::<RigPaletteExtract>();
    assert_eq!(
        old.frame,
        Some(app.world().resource::<bevy::diagnostic::FrameCount>().0)
    );
    assert_eq!(
        old.allocations.iter().filter(|a| a.id != 0).count(),
        1,
        "the late allocation is not in the prior publication"
    );
    assert_eq!(old.allocations[first.0 as usize].bones, 4);
    assert_eq!(late.3, 6);
    assert_ne!(late.1, 0);
    assert_eq!(old.allocations[first.0 as usize].id, first_id);
    assert_eq!(old.allocations[first.0 as usize].base, first.1);
    assert_eq!(old.table[first.0 as usize], first.1);
    assert_eq!(old.table[late.0 as usize], 0);
    assert_eq!(old.watermark, 4);
    assert_eq!(old.allocations[late.0 as usize], RigAllocation::default());
    app.update();
    let next = app.world().resource::<RigPaletteExtract>();
    assert_eq!(next.allocations.iter().filter(|a| a.id != 0).count(), 2);
    assert_eq!(next.allocations[late.0 as usize].id, late.1);
    assert_eq!(next.allocations[late.0 as usize].base, late.2);
    assert_eq!(next.allocations[late.0 as usize].bones, late.3);
    assert_eq!(next.table[late.0 as usize], late.2);
    assert_eq!(next.watermark, 10);
    assert_eq!(
        next.watermark,
        app.world().resource::<RigPalettes>().bone_watermark()
    );
}

fn snapshot_publication(watermark: u32, table_generation: u64) -> RigPaletteExtract {
    RigPaletteExtract {
        rows: Arc::new(vec![[0.0; 4]; 3 * watermark as usize]),
        table: Arc::new(vec![0; MAX_RIG_SLOTS]),
        origins: Arc::new(vec![[0.0; 4]; MAX_RIG_SLOTS]),
        dirty: Arc::new(Vec::new()),
        table_generation,
        frame: Some(1),
        allocations: Arc::new(vec![RigAllocation::default(); MAX_RIG_SLOTS]),
        watermark,
        origin_generation: table_generation,
        origin_mirror_generation: 0,
    }
}

#[test]
fn new_shared_source_reseeds_unchanged_publication() {
    let source = bevy::render::render_resource::BufferId::new();
    let data = snapshot_publication(4, 7);
    let last = UploadedGenerations {
        table: Some(7),
        origin: Some(7),
        shared_source: Some(bevy::render::render_resource::BufferId::new()),
        ..Default::default()
    };
    let plan = shared_rig_upload_plan(&last, source, &data).unwrap();
    assert!(plan.table && plan.origins);
    assert_eq!(plan.ranges, vec![(0, 4)]);
}

#[test]
fn unchanged_shared_source_skips_unchanged_upload() {
    let source = bevy::render::render_resource::BufferId::new();
    let data = snapshot_publication(4, 7);
    let last = UploadedGenerations {
        table: Some(7),
        origin: Some(7),
        dirty: Arc::as_ptr(&data.dirty) as u64,
        shared_source: Some(source),
        ..Default::default()
    };
    let plan = shared_rig_upload_plan(&last, source, &data).unwrap();
    assert!(!plan.table && !plan.origins && plan.ranges.is_empty());
}

#[test]
fn existing_shared_source_preserves_dirty_coalescing() {
    let source = bevy::render::render_resource::BufferId::new();
    let mut data = snapshot_publication(300, 7);
    data.rows = Arc::new(vec![[0.0; 4]; 3 * 604]);
    data.dirty = Arc::new(vec![(0, 4, false), (10, 4, false), (600, 4, false)]);
    let last = UploadedGenerations {
        table: Some(7),
        origin: Some(7),
        dirty: 0,
        shared_source: Some(source),
        ..Default::default()
    };
    let plan = shared_rig_upload_plan(&last, source, &data).unwrap();
    assert_eq!(plan.ranges, vec![(0, 14), (600, 4)]);
    for &(base, len) in &plan.ranges {
        let _ = &data.rows[3 * base as usize..3 * (base + len) as usize];
    }
}

#[test]
fn lost_shared_source_requires_reseed_on_recovery() {
    let source = bevy::render::render_resource::BufferId::new();
    let data = snapshot_publication(4, 7);
    let last = UploadedGenerations {
        table: Some(7),
        origin: Some(7),
        shared_source: None,
        ..Default::default()
    };
    let plan = shared_rig_upload_plan(&last, source, &data).unwrap();
    assert!(plan.table && plan.origins);
    assert_eq!(plan.ranges, vec![(0, 4)]);
}

#[test]
fn empty_palette_source_still_seeds_tables() {
    let source = bevy::render::render_resource::BufferId::new();
    let data = snapshot_publication(0, 3);
    let last = UploadedGenerations::default();
    let plan = shared_rig_upload_plan(&last, source, &data).unwrap();
    assert!(plan.table && plan.origins);
    assert!(plan.ranges.is_empty());
}

#[test]
fn palette_receipt_requires_current_populated_source() {
    let source = bevy::render::render_resource::BufferId::new();
    let data = snapshot_publication(4, 7);
    let mut last = UploadedGenerations {
        table: Some(7),
        origin: Some(7),
        dirty: Arc::as_ptr(&data.dirty) as u64,
        shared_source: Some(source),
        ..Default::default()
    };
    assert_eq!(
        palette_upload_receipt(Some(1), Some(source), Some(&data), &last),
        Some((1, source))
    );
    assert_eq!(
        palette_upload_receipt(None, Some(source), Some(&data), &last),
        None
    );
    assert_eq!(
        palette_upload_receipt(Some(1), None, Some(&data), &last),
        None
    );
    assert_eq!(
        palette_upload_receipt(Some(2), Some(source), Some(&data), &last),
        None
    );
    assert_eq!(
        palette_upload_receipt(
            Some(1),
            Some(bevy::render::render_resource::BufferId::new()),
            Some(&data),
            &last
        ),
        None
    );
    let mut dirty = snapshot_publication(4, 7);
    dirty.dirty = Arc::new(vec![(0, 4, false)]);
    assert_eq!(
        palette_upload_receipt(Some(1), Some(source), Some(&dirty), &last),
        None
    );
    last.table = Some(6);
    assert_eq!(
        palette_upload_receipt(Some(1), Some(source), Some(&data), &last),
        None
    );
    last.table = Some(7);
    last.shared_source = None;
    assert_eq!(
        palette_upload_receipt(Some(1), Some(source), Some(&data), &last),
        None
    );
    last.shared_source = Some(source);
    let mut wrap = snapshot_publication(4, 7);
    wrap.frame = Some(0);
    assert_eq!(
        palette_upload_receipt(Some(0), Some(source), Some(&wrap), &last),
        Some((0, source))
    );
}

#[test]
fn mirror_uploads_continue_without_shared_source() {
    let mut data = snapshot_publication(4, 7);
    data.dirty = Arc::new(vec![(0, 4, true), (10, 4, false), (400, 2, true)]);
    let last = UploadedGenerations {
        table: Some(6),
        origin: Some(6),
        origin_mirror: Some(5),
        dirty: 0,
        ..Default::default()
    };
    let (shared, mirror) = rig_upload_plans(&last, None, &data);
    assert!(shared.is_none());
    assert!(mirror.table && mirror.origins);
    assert_eq!(mirror.ranges, vec![(0, 4), (400, 2)]);
}

#[test]
fn mirror_uploads_do_not_require_shared_allocation_metadata() {
    let source = bevy::render::render_resource::BufferId::new();
    let mut data = snapshot_publication(4, 7);
    data.allocations = Arc::new(Vec::new());
    let last = UploadedGenerations::default();
    let (shared, mirror) = rig_upload_plans(&last, Some(source), &data);
    assert!(shared.is_none());
    assert!(mirror.table && mirror.origins);
}

#[test]
fn unchanged_mirrors_request_no_writes() {
    let data = snapshot_publication(4, 7);
    let last = UploadedGenerations {
        table: Some(7),
        origin: Some(7),
        origin_mirror: Some(0),
        dirty: Arc::as_ptr(&data.dirty) as u64,
        ..Default::default()
    };
    let (_, mirror) = rig_upload_plans(&last, None, &data);
    assert!(!mirror.table && !mirror.origins && mirror.ranges.is_empty());
}

#[test]
fn palette_receipt_rejects_incomplete_shapes_and_advances_epochs() {
    let source = bevy::render::render_resource::BufferId::new();
    let mut data = snapshot_publication(4, 7);
    let last = UploadedGenerations {
        table: Some(7),
        origin: Some(7),
        dirty: Arc::as_ptr(&data.dirty) as u64,
        shared_source: Some(source),
        ..Default::default()
    };
    assert_eq!(
        palette_upload_receipt(Some(1), Some(source), None, &last),
        None
    );
    assert_eq!(
        palette_upload_receipt(Some(1), Some(source), Some(&data), &last),
        Some((1, source))
    );

    let mut stale = data.clone();
    stale.origin_generation = 6;
    assert_eq!(
        palette_upload_receipt(Some(1), Some(source), Some(&stale), &last),
        None
    );
    let mut short = data.clone();
    short.table = Arc::new(vec![0; MAX_RIG_SLOTS - 1]);
    assert_eq!(
        palette_upload_receipt(Some(1), Some(source), Some(&short), &last),
        None
    );
    let mut short = data.clone();
    short.origins = Arc::new(vec![[0.0; 4]; MAX_RIG_SLOTS - 1]);
    assert_eq!(
        palette_upload_receipt(Some(1), Some(source), Some(&short), &last),
        None
    );
    let mut short = data.clone();
    short.allocations = Arc::new(vec![RigAllocation::default(); MAX_RIG_SLOTS - 1]);
    assert_eq!(
        palette_upload_receipt(Some(1), Some(source), Some(&short), &last),
        None
    );
    let mut short = data.clone();
    short.rows = Arc::new(vec![[0.0; 4]; 11]);
    assert_eq!(
        palette_upload_receipt(Some(1), Some(source), Some(&short), &last),
        None
    );
    let mut high = data.clone();
    high.watermark = MAX_PALETTE_BONES as u32 + 1;
    assert_eq!(
        palette_upload_receipt(Some(1), Some(source), Some(&high), &last),
        None
    );

    data.frame = Some(2);
    assert_eq!(
        palette_upload_receipt(Some(2), Some(source), Some(&data), &last),
        Some((2, source))
    );
    data.frame = Some(u32::MAX);
    assert_eq!(
        palette_upload_receipt(Some(u32::MAX), Some(source), Some(&data), &last),
        Some((u32::MAX, source))
    );
    data.frame = Some(0);
    assert_eq!(
        palette_upload_receipt(Some(0), Some(source), Some(&data), &last),
        Some((0, source))
    );
}
