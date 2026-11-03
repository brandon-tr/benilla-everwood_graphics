//! The previous-rig planner, the render-world state machine and the production schedule order.

use super::*;

fn allocations() -> Vec<RigAllocation> {
    vec![RigAllocation::default(); MAX_RIG_SLOTS]
}

fn slot(id: u64, base: u32, bones: u32) -> RigAllocation {
    RigAllocation { id, base, bones }
}

fn epoch(
    frame: u32,
    source: BufferId,
    destination: BufferId,
    alloc: &[RigAllocation],
    watermark: u32,
) -> RigTaaEpochView {
    RigTaaEpochView {
        frame,
        source,
        destination,
        allocations: Arc::new(alloc.to_vec()),
        watermark,
    }
}

fn confirmed(epoch: &RigTaaEpochView) -> RigTaaRecordInput {
    RigTaaRecordInput {
        epoch: epoch.clone(),
        light: Some((epoch.frame, epoch.source)),
        palette: Some((epoch.frame, epoch.source)),
    }
}

/// One frame's stages with the epoch's own destination bound, and every effect acknowledged to the
/// buffer the write actually targeted (as the systems do when the target is present and safe).
fn run(
    state: &mut RigTaaSnapshotState,
    demand: bool,
    epoch: &RigTaaEpochView,
    receipts: &RigTaaRecordInput,
) -> (RigTaaCopyAction, RigTaaRecordAction) {
    run_frame(
        state,
        Some(epoch.destination),
        demand,
        Some(epoch),
        Some(receipts),
        true,
    )
}

/// One frame's stages with explicit inputs: which destination is actually bound, whether it can
/// take a clear, and whether demand/epoch/receipts are available.
fn run_frame(
    state: &mut RigTaaSnapshotState,
    observed: Option<BufferId>,
    demand: bool,
    epoch: Option<&RigTaaEpochView>,
    receipts: Option<&RigTaaRecordInput>,
    clearable: bool,
) -> (RigTaaCopyAction, RigTaaRecordAction) {
    let copy = copy_stage(state, observed, demand, epoch);
    if clearable {
        acknowledge_clear(state, observed, &copy);
    }
    let record = record_stage(state, observed, demand, receipts);
    if clearable {
        acknowledge_record(state, observed, &record);
    } else if let RigTaaRecordAction::Publish(_) = record {
        // Metadata can still be queued when only the *failure-path* clear is unavailable.
        if let Some(destination) = observed {
            state.metadata_queued(destination);
        }
    }
    (copy, record)
}

/// The clear acknowledgement: the queued write went to `observed`, and only that buffer's
/// obligation is discharged.
fn acknowledge_clear(
    state: &mut RigTaaSnapshotState,
    observed: Option<BufferId>,
    copy: &RigTaaCopyAction,
) {
    if *copy == RigTaaCopyAction::Invalidate {
        if let Some(destination) = observed {
            state.clear_queued(destination);
        }
    }
}

/// The record system's acknowledgement: queued confirmed metadata advertises a valid snapshot to
/// its own destination, a queued clear discharges that destination's invalidation obligation.
fn acknowledge_record(
    state: &mut RigTaaSnapshotState,
    observed: Option<BufferId>,
    record: &RigTaaRecordAction,
) {
    match record {
        RigTaaRecordAction::Publish(_) => {
            if let Some(destination) = observed {
                state.metadata_queued(destination);
            }
        }
        RigTaaRecordAction::Invalidate => {
            if let Some(destination) = observed {
                state.clear_queued(destination);
            }
        }
        RigTaaRecordAction::Idle => {}
    }
}

/// Both systems' acknowledgement for one frame, when the destination is safe to write.
fn acknowledge_queued(
    state: &mut RigTaaSnapshotState,
    observed: Option<BufferId>,
    copy: &RigTaaCopyAction,
    record: &RigTaaRecordAction,
) {
    acknowledge_clear(state, observed, copy);
    acknowledge_record(state, observed, record);
}

/// One frame's stages with no destination available to queue a clear: the invalidation obligation
/// must survive exactly as it does when `RigTaaBinds` is gone.
fn run_without_clear_target(
    state: &mut RigTaaSnapshotState,
    demand: bool,
    epoch: Option<&RigTaaEpochView>,
    receipts: Option<&RigTaaRecordInput>,
) -> (RigTaaCopyAction, RigTaaRecordAction) {
    run_frame(state, None, demand, epoch, receipts, false)
}

/// A state holding one confirmed epoch and no published snapshot: exactly a first demand frame.
/// The first observation adopts the destination and clears it before anything else.
fn seeded(
    source: BufferId,
    destination: BufferId,
    alloc: &[RigAllocation],
    watermark: u32,
) -> (RigTaaSnapshotState, RigTaaEpochView) {
    let mut state = RigTaaSnapshotState::default();
    let first = epoch(1, source, destination, alloc, watermark);
    let (copy, record) = run(&mut state, true, &first, &confirmed(&first));
    assert_eq!(
        copy,
        RigTaaCopyAction::Invalidate,
        "a newly adopted destination is cleared first"
    );
    assert_eq!(record, RigTaaRecordAction::Idle);
    assert!(
        !state.header_valid,
        "the adoption clear discharged the obligation"
    );
    (state, first)
}

/// A state holding a published snapshot for epoch 2.
fn published(
    source: BufferId,
    destination: BufferId,
    alloc: &[RigAllocation],
    watermark: u32,
) -> RigTaaSnapshotState {
    let (mut state, _) = seeded(source, destination, alloc, watermark);
    let second = epoch(2, source, destination, alloc, watermark);
    let (copy, record) = run(&mut state, true, &second, &confirmed(&second));
    assert!(matches!(copy, RigTaaCopyAction::Submit(_)));
    assert!(matches!(record, RigTaaRecordAction::Publish(_)));
    assert!(state.header_valid);
    state
}

/// One frame that only records the epoch (there is no previous side to copy, and the destination
/// is already adopted).
fn seed(state: &mut RigTaaSnapshotState, epoch: &RigTaaEpochView) {
    assert_eq!(copy_frame(state, true, Some(epoch)), RigTaaCopyAction::Idle);
    assert_eq!(
        record_frame(state, true, Some(&confirmed(epoch))),
        RigTaaRecordAction::Idle
    );
}

/// One frame that rotates and confirms, publishing a valid snapshot.
fn rotate(state: &mut RigTaaSnapshotState, epoch: &RigTaaEpochView) {
    let copy = copy_frame(state, true, Some(epoch));
    assert!(matches!(copy, RigTaaCopyAction::Submit(_)));
    let record = record_frame(state, true, Some(&confirmed(epoch)));
    assert!(matches!(record, RigTaaRecordAction::Publish(_)));
    acknowledge_queued(state, Some(epoch.destination), &copy, &record);
    assert!(state.header_valid);
}

/// One copy-stage call with the epoch's own destination bound (the common test shape).
fn copy_frame(
    state: &mut RigTaaSnapshotState,
    demand: bool,
    epoch: Option<&RigTaaEpochView>,
) -> RigTaaCopyAction {
    copy_stage(state, epoch.map(|epoch| epoch.destination), demand, epoch)
}

/// One record-stage call with the receipts' own destination bound.
fn record_frame(
    state: &mut RigTaaSnapshotState,
    demand: bool,
    receipts: Option<&RigTaaRecordInput>,
) -> RigTaaRecordAction {
    record_stage(
        state,
        receipts.map(|receipts| receipts.epoch.destination),
        demand,
        receipts,
    )
}

/// Acknowledge both effects to the destination the state currently recognises: exactly the buffer
/// the test's own stage call observed and the systems would have written.
fn acknowledge_queued_now(
    state: &mut RigTaaSnapshotState,
    copy: &RigTaaCopyAction,
    record: &RigTaaRecordAction,
) {
    let destination = state.destination;
    acknowledge_queued(state, destination, copy, record);
}

/// Acknowledge a clear to the destination the state currently recognises: exactly the buffer the
/// test's own stage call observed and the system would have written.
fn ack_clear_now(state: &mut RigTaaSnapshotState) {
    let destination = state
        .destination
        .expect("a destination was observed by the stage call");
    state.clear_queued(destination);
}

/// One copy-stage call with no complete epoch view, but a destination still bound.
fn copy_frame_without_epoch(
    state: &mut RigTaaSnapshotState,
    observed: Option<BufferId>,
    demand: bool,
) -> RigTaaCopyAction {
    copy_stage(state, observed, demand, None)
}

/// One record-stage call with no complete epoch view, but a destination still bound.
fn record_frame_without_epoch(
    state: &mut RigTaaSnapshotState,
    observed: Option<BufferId>,
    demand: bool,
) -> RigTaaRecordAction {
    record_stage(state, observed, demand, None)
}

// ---------------------------------------------------------------- the copy planner

#[test]
fn consecutive_and_wrapping_epochs_use_prior_upload() {
    let source = BufferId::new();
    let mut alloc = allocations();
    alloc[1] = slot(7, 0, 4);
    let previous = RigTaaUploadedEpoch {
        frame: 41,
        source,
        allocations: Arc::new(alloc.clone()),
        watermark: 4,
    };
    let plan = plan_previous_rig_copy(&previous, 42, source, &alloc).expect("consecutive");
    assert_eq!(plan.header, [1, 41, 42, 4]);
    assert_eq!(plan.slots[1], [1, 7, 0, 4]);

    let previous = RigTaaUploadedEpoch {
        frame: u32::MAX,
        source,
        allocations: Arc::new(alloc.clone()),
        watermark: 4,
    };
    let plan = plan_previous_rig_copy(&previous, 0, source, &alloc).expect("wrapped frame zero");
    assert_eq!(plan.header[1..3], [u32::MAX, 0]);
}

#[test]
fn gaps_and_source_or_destination_changes_invalidate() {
    let source = BufferId::new();
    let alloc = allocations();
    let previous = RigTaaUploadedEpoch {
        frame: 10,
        source,
        allocations: Arc::new(alloc.clone()),
        watermark: 0,
    };
    assert!(plan_previous_rig_copy(&previous, 12, source, &alloc).is_none());
    assert!(plan_previous_rig_copy(&previous, 11, BufferId::new(), &alloc).is_none());
    // An incomplete slot table is not a rotation either.
    assert!(plan_previous_rig_copy(&previous, 11, source, &alloc[..MAX_RIG_SLOTS - 1]).is_none());
    let mut short = previous.clone();
    short.allocations = Arc::new(alloc[..MAX_RIG_SLOTS - 1].to_vec());
    assert!(plan_previous_rig_copy(&short, 11, source, &alloc).is_none());
}

#[test]
fn slot_reuse_free_and_changed_ranges_zero_metadata() {
    let source = BufferId::new();
    let mut previous_alloc = allocations();
    previous_alloc[1] = slot(7, 0, 4);
    previous_alloc[2] = slot(8, 4, 4);
    previous_alloc[3] = slot(9, 8, 4);
    let mut current = previous_alloc.clone();
    current[1] = slot(10, 0, 4);
    current[2] = RigAllocation::default();
    current[3] = slot(9, 8, 5);
    let previous = RigTaaUploadedEpoch {
        frame: 1,
        source,
        allocations: Arc::new(previous_alloc),
        watermark: 12,
    };
    let plan = plan_previous_rig_copy(&previous, 2, source, &current).unwrap();
    assert_eq!(plan.slots[1], [0; 4]);
    assert_eq!(plan.slots[2], [0; 4]);
    assert_eq!(plan.slots[3], [0; 4]);
}

#[test]
fn a_valid_unrelated_slot_survives_its_neighbours_rejection() {
    let source = BufferId::new();
    let mut previous_alloc = allocations();
    previous_alloc[1] = slot(7, 0, 4); // unchanged: stays valid
    previous_alloc[2] = slot(8, 4, 4); // reused this epoch
    previous_alloc[3] = slot(9, 8, 4); // freed this epoch
    let mut current = previous_alloc.clone();
    current[2] = slot(10, 4, 4);
    current[3] = RigAllocation::default();
    let previous = RigTaaUploadedEpoch {
        frame: 1,
        source,
        allocations: Arc::new(previous_alloc),
        watermark: 12,
    };
    let plan = plan_previous_rig_copy(&previous, 2, source, &current).unwrap();
    assert_eq!(plan.slots[1], [1, 7, 0, 4]);
    assert_eq!(plan.slots[2], [0; 4]);
    assert_eq!(plan.slots[3], [0; 4]);
    assert_eq!(plan.slots[0], [0; 4], "slot 0 is the no-rig sentinel");
    assert_eq!(
        plan.slots.iter().filter(|slot| slot[0] == 1).count(),
        1,
        "exactly the one eligible slot is advertised"
    );
}

/// The full 64-bit allocation identity is compared and published in both 32-bit halves; the high
/// word alone is enough to make two otherwise identical allocations different.
#[test]
fn the_planner_compares_and_publishes_the_full_allocation_identity() {
    let source = BufferId::new();
    let mut previous_alloc = allocations();
    previous_alloc[1] = slot(0x0000_0003_0000_0007, 0, 4);
    previous_alloc[2] = slot(0x0000_0003_0000_0008, 4, 4);
    let mut current = previous_alloc.clone();
    // Same low word, different high word: a different allocation.
    current[1] = slot(0x0000_0004_0000_0007, 0, 4);
    // Same high word, different low word: also different.
    current[2] = slot(0x0000_0003_0000_0009, 4, 4);
    let previous = RigTaaUploadedEpoch {
        frame: 1,
        source,
        allocations: Arc::new(previous_alloc),
        watermark: 8,
    };
    let plan = plan_previous_rig_copy(&previous, 2, source, &current).unwrap();
    assert_eq!(plan.slots[1], [0; 4]);
    assert_eq!(plan.slots[2], [0; 4]);

    let mut current = allocations();
    current[3] = slot(0x0000_0003_0000_0007, 0, 4);
    let mut previous_alloc = allocations();
    previous_alloc[3] = slot(0x0000_0003_0000_0007, 0, 4);
    let previous = RigTaaUploadedEpoch {
        frame: 1,
        source,
        allocations: Arc::new(previous_alloc),
        watermark: 4,
    };
    let plan = plan_previous_rig_copy(&previous, 2, source, &current).unwrap();
    assert_eq!(
        plan.slots[3],
        [1, 7, 3, 4],
        "low word, high word, bone count"
    );
}

#[test]
fn a_slot_needs_a_positive_equal_bone_count_an_equal_base_and_a_valid_range() {
    let source = BufferId::new();
    let mut previous_alloc = allocations();
    previous_alloc[1] = slot(7, 0, 0); // zero bones, nonzero identity: not a rig
    previous_alloc[2] = slot(8, 4, 4); // base moved
    previous_alloc[3] = slot(9, 13, 4); // one bone past the watermark
    previous_alloc[4] = slot(10, 12, 4); // exactly the watermark: covered
    previous_alloc[5] = slot(11, u32::MAX - 1, 4); // range overflow
    let mut current = previous_alloc.clone();
    current[2] = slot(8, 6, 4);
    let previous = RigTaaUploadedEpoch {
        frame: 1,
        source,
        allocations: Arc::new(previous_alloc),
        watermark: 16,
    };
    let plan = plan_previous_rig_copy(&previous, 2, source, &current).unwrap();
    assert_eq!(plan.slots[1], [0; 4], "zero bones");
    assert_eq!(plan.slots[2], [0; 4], "moved base");
    assert_eq!(plan.slots[3], [0; 4], "range past the copied prefix");
    assert_eq!(
        plan.slots[4],
        [1, 10, 0, 4],
        "range ending at the watermark"
    );
    assert_eq!(plan.slots[5], [0; 4], "range overflow");
}

#[test]
fn copy_regions_use_previous_watermark_and_complete_monkey_frame() {
    let source = BufferId::new();
    let alloc = allocations();
    let previous = RigTaaUploadedEpoch {
        frame: 3,
        source,
        allocations: Arc::new(alloc.clone()),
        watermark: 0,
    };
    let plan = plan_previous_rig_copy(&previous, 4, source, &alloc).unwrap();
    assert!(plan
        .regions
        .iter()
        .all(|r| r.destination_offset != PALETTE_OFFSET));
    let monkey = plan
        .regions
        .iter()
        .find(|r| r.destination_offset == HEADER_BYTES)
        .unwrap();
    assert_eq!(
        (monkey.bytes, monkey.source_offset),
        (
            MONKEY_BYTES,
            crate::lighting::per_frame_blob_bytes() - MONKEY_BYTES
        )
    );

    let previous = RigTaaUploadedEpoch {
        frame: 4,
        source,
        allocations: Arc::new(alloc.clone()),
        watermark: 4,
    };
    let plan = plan_previous_rig_copy(&previous, 5, source, &alloc).unwrap();
    let palette = plan
        .regions
        .iter()
        .find(|r| r.destination_offset == PALETTE_OFFSET)
        .unwrap();
    assert_eq!(palette.bytes, 4 * BONE_BYTES);
}

/// Every region is exactly where the declared layout puts it, and nothing reads or writes outside
/// the two buffers.
#[test]
fn the_copy_plan_is_exactly_the_declared_layout() {
    let source = BufferId::new();
    let alloc = allocations();
    let previous = RigTaaUploadedEpoch {
        frame: 1,
        source,
        allocations: Arc::new(alloc.clone()),
        watermark: 8,
    };
    let plan = plan_previous_rig_copy(&previous, 2, source, &alloc).unwrap();
    assert_eq!(
        plan.regions,
        vec![
            RigTaaCopy {
                source_offset: crate::lighting::per_frame_blob_bytes() - MONKEY_BYTES,
                destination_offset: HEADER_BYTES,
                bytes: MONKEY_BYTES,
            },
            RigTaaCopy {
                source_offset: crate::rig_palette::rig_table_region_offset(),
                destination_offset: SLOT_BASES_OFFSET,
                bytes: SLOT_BASES_BYTES,
            },
            RigTaaCopy {
                source_offset: crate::rig_palette::rig_origin_region_offset(),
                destination_offset: SLOT_ORIGINS_OFFSET,
                bytes: SLOT_ORIGINS_BYTES,
            },
            RigTaaCopy {
                source_offset: crate::rig_palette::palette_region_offset(),
                destination_offset: PALETTE_OFFSET,
                bytes: 8 * BONE_BYTES,
            },
        ]
    );
    assert_eq!(plan.source_extent(), copied_source_extent(8));
    assert_eq!(plan.destination_extent(), PALETTE_OFFSET + 8 * BONE_BYTES);
    assert!(plan.destination_extent() <= PREVIOUS_RIG_BYTES);
    assert_eq!(
        plan.destination_extent(),
        plan.destination_extent().next_multiple_of(4),
        "wgpu copy offsets and sizes are 4-byte aligned"
    );

    // A watermark just inside the palette slab still fits the fixed destination.
    let previous = RigTaaUploadedEpoch {
        frame: 1,
        source,
        allocations: Arc::new(alloc.clone()),
        watermark: MAX_PALETTE_BONES as u32,
    };
    let plan = plan_previous_rig_copy(&previous, 2, source, &alloc).unwrap();
    assert_eq!(plan.destination_extent(), PREVIOUS_RIG_BYTES);
    assert_eq!(copied_source_extent(MAX_PALETTE_BONES as u32), {
        crate::rig_palette::palette_region_offset() + PALETTE_BYTES
    });
}

#[test]
fn the_metadata_abi_is_the_proposed_one() {
    let source = BufferId::new();
    let mut alloc = allocations();
    alloc[1] = slot(7, 0, 4);
    let previous = RigTaaUploadedEpoch {
        frame: 9,
        source,
        allocations: Arc::new(alloc.clone()),
        watermark: 4,
    };
    let plan = plan_previous_rig_copy(&previous, 10, source, &alloc).unwrap();
    assert_eq!(
        plan.header,
        [1, 9, 10, 4],
        "valid, previous, current, watermark"
    );
    assert_eq!(plan.slots.len(), MAX_RIG_SLOTS);
    assert_eq!(plan.slots[1], [1, 7, 0, 4]);
}

// ---------------------------------------------------------------- buffer usability

#[test]
fn the_two_buffers_must_be_distinct_usable_and_big_enough() {
    let source_id = BufferId::new();
    let destination_id = BufferId::new();
    let source = RigTaaBufferFacts {
        id: source_id,
        size: copied_source_extent(8),
        copy_src: true,
        copy_dst: false,
    };
    let destination = RigTaaBufferFacts {
        id: destination_id,
        size: PREVIOUS_RIG_BYTES,
        copy_src: false,
        copy_dst: true,
    };
    assert!(snapshot_buffers_usable(source, destination, 8));
    assert!(
        !snapshot_buffers_usable(source, destination, 9),
        "one bone past the source extent"
    );
    // Aliasing alone: the same identity on an otherwise fully valid pair (COPY_SRC and COPY_DST,
    // and big enough on both sides), so no other condition can be the reason for the rejection.
    let alias = RigTaaBufferFacts {
        id: source_id,
        size: PREVIOUS_RIG_BYTES,
        copy_src: true,
        copy_dst: true,
    };
    assert!(
        !snapshot_buffers_usable(alias, alias, 8),
        "source/destination aliasing with every other condition satisfied"
    );
    assert!(
        !snapshot_buffers_usable(
            RigTaaBufferFacts {
                copy_src: false,
                ..source
            },
            destination,
            8
        ),
        "source without COPY_SRC"
    );
    assert!(
        !snapshot_buffers_usable(
            source,
            RigTaaBufferFacts {
                copy_dst: false,
                ..destination
            },
            8
        ),
        "destination without COPY_DST"
    );
    assert!(
        !snapshot_buffers_usable(
            RigTaaBufferFacts {
                size: copied_source_extent(8) - 4,
                ..source
            },
            destination,
            8
        ),
        "short source"
    );
    assert!(
        !snapshot_buffers_usable(
            source,
            RigTaaBufferFacts {
                size: PREVIOUS_RIG_BYTES - 16,
                ..destination
            },
            8
        ),
        "short destination"
    );
}

// ---------------------------------------------------------------- the state machine

#[test]
fn first_demand_seeds_invalid_and_the_next_consecutive_epoch_publishes() {
    let source = BufferId::new();
    let destination = BufferId::new();
    let alloc = allocations();
    let (mut state, _) = seeded(source, destination, &alloc, 0);
    assert_eq!(state.previous.as_ref().unwrap().frame, 1);
    assert_eq!(state.previous.as_ref().unwrap().source, source);
    assert!(!state.header_valid, "the seed frame publishes nothing");
    assert!(state.published.is_none());

    let second = epoch(2, source, destination, &alloc, 0);
    let (copy, record) = run(&mut state, true, &second, &confirmed(&second));
    let RigTaaCopyAction::Submit(plan) = copy else {
        panic!("a consecutive receipt-backed epoch copies: {copy:?}");
    };
    assert_eq!(plan.header, [1, 1, 2, 0]);
    let RigTaaRecordAction::Publish(metadata) = record else {
        panic!("the confirmed epoch publishes its metadata: {record:?}");
    };
    assert_eq!(metadata.header, [1, 1, 2, 0]);
    assert_eq!(metadata.header, plan.header);
    assert!(state.header_valid);
    assert_eq!(state.published.as_ref().map(|epoch| epoch.frame), Some(2));
    assert_eq!(state.previous.as_ref().unwrap().frame, 2);

    let third = epoch(3, source, destination, &alloc, 0);
    let (copy, record) = run(&mut state, true, &third, &confirmed(&third));
    assert!(matches!(copy, RigTaaCopyAction::Submit(_)));
    assert!(matches!(record, RigTaaRecordAction::Publish(_)));
    assert_eq!(state.published.as_ref().map(|epoch| epoch.frame), Some(3));
}

#[test]
fn no_demand_invalidates_the_header_once_and_retires_the_record() {
    let source = BufferId::new();
    let destination = BufferId::new();
    let alloc = allocations();
    let mut state = published(source, destination, &alloc, 0);

    let next = epoch(3, source, destination, &alloc, 0);
    let (copy, record) = run(&mut state, false, &next, &confirmed(&next));
    assert_eq!(copy, RigTaaCopyAction::Invalidate, "one invalidation write");
    assert_eq!(
        record,
        RigTaaRecordAction::Idle,
        "the header is already invalid, so no second write"
    );
    assert!(state.previous.is_none());
    assert!(!state.header_valid);

    // Still off: nothing is submitted and nothing is written again.
    let later = epoch(4, source, destination, &alloc, 0);
    let (copy, record) = run(&mut state, false, &later, &confirmed(&later));
    assert_eq!(copy, RigTaaCopyAction::Idle);
    assert_eq!(record, RigTaaRecordAction::Idle);

    // Demand returning starts from a seed frame, not from the retired epoch.
    let resumed = epoch(5, source, destination, &alloc, 0);
    assert_eq!(
        copy_frame(&mut state, true, Some(&resumed)),
        RigTaaCopyAction::Idle
    );
    assert_eq!(
        record_frame(&mut state, true, Some(&confirmed(&resumed))),
        RigTaaRecordAction::Idle
    );
    assert_eq!(state.previous.as_ref().unwrap().frame, 5);
}

#[test]
fn duplicate_invocations_never_rotate_or_republish() {
    let source = BufferId::new();
    let destination = BufferId::new();
    let alloc = allocations();
    let mut state = published(source, destination, &alloc, 0);

    let third = epoch(3, source, destination, &alloc, 0);
    let copy = copy_frame(&mut state, true, Some(&third));
    assert!(matches!(copy, RigTaaCopyAction::Submit(_)));
    assert_eq!(
        copy_frame(&mut state, true, Some(&third)),
        RigTaaCopyAction::Idle,
        "a duplicate copy stage must not rotate twice"
    );
    let record = record_frame(&mut state, true, Some(&confirmed(&third)));
    assert!(matches!(record, RigTaaRecordAction::Publish(_)));
    acknowledge_queued_now(&mut state, &copy, &record);
    assert_eq!(
        record_frame(&mut state, true, Some(&confirmed(&third))),
        RigTaaRecordAction::Idle,
        "a duplicate record republishes nothing"
    );
    let (copy, record) = run(&mut state, true, &third, &confirmed(&third));
    assert_eq!(copy, RigTaaCopyAction::Idle);
    assert_eq!(record, RigTaaRecordAction::Idle);
    assert!(state.header_valid);
}

#[test]
fn loss_on_a_duplicate_still_invalidates_and_recovery_cannot_label_current_bytes() {
    let source = BufferId::new();
    let destination = BufferId::new();
    let alloc = allocations();
    let mut state = published(source, destination, &alloc, 0);

    let third = epoch(3, source, destination, &alloc, 0);
    assert!(matches!(
        copy_frame(&mut state, true, Some(&third)),
        RigTaaCopyAction::Submit(_)
    ));
    // The duplicate loses its destination: the stage owes a clear it cannot queue, so the
    // obligation must survive rather than being reported as done.
    assert_eq!(
        copy_frame(&mut state, true, None),
        RigTaaCopyAction::Invalidate
    );
    assert!(state.previous.is_none());
    assert!(state.header_valid, "the clear is still owed");

    // Recovery inside the same epoch: the destination is back, so the clear is queued first...
    assert_eq!(
        copy_frame(&mut state, true, Some(&third)),
        RigTaaCopyAction::Invalidate
    );
    ack_clear_now(&mut state);
    assert!(!state.header_valid);

    // ...and the epoch must neither re-copy nor publish metadata for the invalidated copy.
    assert_eq!(
        copy_frame(&mut state, true, Some(&third)),
        RigTaaCopyAction::Idle
    );
    assert_eq!(
        record_frame(&mut state, true, Some(&confirmed(&third))),
        RigTaaRecordAction::Idle
    );
    assert_eq!(state.previous.as_ref().unwrap().frame, 3);
    assert!(!state.header_valid);

    // Consecutive epochs rotate again once seeded.
    let fourth = epoch(4, source, destination, &alloc, 0);
    let copy = copy_frame(&mut state, true, Some(&fourth));
    assert!(matches!(copy, RigTaaCopyAction::Submit(_)));
    let record = record_frame(&mut state, true, Some(&confirmed(&fourth)));
    assert!(matches!(record, RigTaaRecordAction::Publish(_)));
    acknowledge_queued_now(&mut state, &copy, &record);
    assert!(state.header_valid);
}

#[test]
fn demand_loss_on_a_duplicate_still_invalidates() {
    let source = BufferId::new();
    let destination = BufferId::new();
    let alloc = allocations();
    let mut state = published(source, destination, &alloc, 0);

    let third = epoch(3, source, destination, &alloc, 0);
    assert!(matches!(
        copy_frame(&mut state, true, Some(&third)),
        RigTaaCopyAction::Submit(_)
    ));
    assert_eq!(
        copy_frame(&mut state, false, Some(&third)),
        RigTaaCopyAction::Invalidate
    );
    assert!(state.previous.is_none());
    ack_clear_now(&mut state);
    assert!(!state.header_valid);
    assert_eq!(
        record_frame(&mut state, false, Some(&confirmed(&third))),
        RigTaaRecordAction::Idle
    );
}

#[test]
fn missing_or_mismatched_receipts_invalidate_the_record() {
    let source = BufferId::new();
    let destination = BufferId::new();
    let alloc = allocations();
    let mut state = published(source, destination, &alloc, 0);

    // A stale light receipt drops the valid header.
    let third = epoch(3, source, destination, &alloc, 0);
    assert!(matches!(
        copy_frame(&mut state, true, Some(&third)),
        RigTaaCopyAction::Submit(_)
    ));
    let mut stale_light = confirmed(&third);
    stale_light.light = Some((2, third.source));
    assert_eq!(
        record_frame(&mut state, true, Some(&stale_light)),
        RigTaaRecordAction::Invalidate
    );
    assert!(state.previous.is_none());
    assert!(state.header_valid, "the clear is owed until it is queued");
    ack_clear_now(&mut state);
    assert!(!state.header_valid);

    // Re-seed, publish, then a palette receipt from another source drops the header.
    seed(&mut state, &epoch(4, source, destination, &alloc, 0));
    rotate(&mut state, &epoch(5, source, destination, &alloc, 0));
    let sixth = epoch(6, source, destination, &alloc, 0);
    assert!(matches!(
        copy_frame(&mut state, true, Some(&sixth)),
        RigTaaCopyAction::Submit(_)
    ));
    let mut wrong_source = confirmed(&sixth);
    wrong_source.palette = Some((sixth.frame, BufferId::new()));
    assert_eq!(
        record_frame(&mut state, true, Some(&wrong_source)),
        RigTaaRecordAction::Invalidate
    );
    assert!(state.previous.is_none());
    assert!(state.header_valid, "the clear is owed until it is queued");
    ack_clear_now(&mut state);
    assert!(!state.header_valid);

    // ...and a missing receipt is just as invalid.
    seed(&mut state, &epoch(7, source, destination, &alloc, 0));
    rotate(&mut state, &epoch(8, source, destination, &alloc, 0));
    let ninth = epoch(9, source, destination, &alloc, 0);
    assert!(matches!(
        copy_frame(&mut state, true, Some(&ninth)),
        RigTaaCopyAction::Submit(_)
    ));
    let mut missing = confirmed(&ninth);
    missing.palette = None;
    assert_eq!(
        record_frame(&mut state, true, Some(&missing)),
        RigTaaRecordAction::Invalidate,
        "a missing receipt drops the header that advertised the copy"
    );
    assert!(state.previous.is_none());
    assert!(state.header_valid, "the clear is owed until it is queued");
    ack_clear_now(&mut state);
    assert!(!state.header_valid);
}
#[test]
fn a_gap_reseeds_instead_of_copying() {
    let source = BufferId::new();
    let destination = BufferId::new();
    let alloc = allocations();
    let mut state = published(source, destination, &alloc, 0);

    // Frame 3 never reached this service: frame 4 cannot claim it was the previous epoch.
    let fourth = epoch(4, source, destination, &alloc, 0);
    assert_eq!(
        copy_frame(&mut state, true, Some(&fourth)),
        RigTaaCopyAction::Invalidate
    );
    ack_clear_now(&mut state);
    assert!(state.previous.is_none());
    assert!(matches!(
        copy_frame(&mut state, true, Some(&fourth)),
        RigTaaCopyAction::Idle
    ));
    assert_eq!(
        record_frame(&mut state, true, Some(&confirmed(&fourth))),
        RigTaaRecordAction::Idle
    );
    assert_eq!(state.previous.as_ref().unwrap().frame, 4);

    let fifth = epoch(5, source, destination, &alloc, 0);
    assert!(matches!(
        copy_frame(&mut state, true, Some(&fifth)),
        RigTaaCopyAction::Submit(_)
    ));
    assert!(matches!(
        record_frame(&mut state, true, Some(&confirmed(&fifth))),
        RigTaaRecordAction::Publish(_)
    ));
}

#[test]
fn frame_zero_after_the_u32_wrap_is_a_valid_snapshot() {
    let source = BufferId::new();
    let destination = BufferId::new();
    let alloc = allocations();
    let mut state = RigTaaSnapshotState::default();
    let last = epoch(u32::MAX, source, destination, &alloc, 2);
    assert_eq!(
        run(&mut state, true, &last, &confirmed(&last)),
        (RigTaaCopyAction::Invalidate, RigTaaRecordAction::Idle),
        "the first observation adopts and clears the destination"
    );
    assert_eq!(state.previous.as_ref().unwrap().frame, u32::MAX);

    let zero = epoch(0, source, destination, &alloc, 2);
    let (copy, record) = run(&mut state, true, &zero, &confirmed(&zero));
    let RigTaaCopyAction::Submit(plan) = copy else {
        panic!("frame zero follows u32::MAX: {copy:?}");
    };
    assert_eq!(plan.header, [1, u32::MAX, 0, 2]);
    let RigTaaRecordAction::Publish(metadata) = record else {
        panic!("frame zero is a real epoch, not a sentinel: {record:?}");
    };
    assert_eq!(metadata.header, [1, u32::MAX, 0, 2]);
    assert!(state.header_valid);
    assert_eq!(state.published.as_ref().map(|epoch| epoch.frame), Some(0));
}

#[test]
fn source_replacement_reseeds_before_it_can_copy() {
    let source = BufferId::new();
    let destination = BufferId::new();
    let alloc = allocations();
    let mut state = published(source, destination, &alloc, 0);

    let replaced = BufferId::new();
    let third = epoch(3, replaced, destination, &alloc, 0);
    assert_eq!(
        copy_frame(&mut state, true, Some(&third)),
        RigTaaCopyAction::Invalidate,
        "a new source buffer has no previous contents to copy"
    );
    ack_clear_now(&mut state);
    assert!(state.previous.is_none());
    assert_eq!(
        record_frame(&mut state, true, Some(&confirmed(&third))),
        RigTaaRecordAction::Idle
    );
    assert_eq!(state.previous.as_ref().unwrap().source, replaced);

    let fourth = epoch(4, replaced, destination, &alloc, 0);
    assert!(matches!(
        copy_frame(&mut state, true, Some(&fourth)),
        RigTaaCopyAction::Submit(_)
    ));
    assert!(matches!(
        record_frame(&mut state, true, Some(&confirmed(&fourth))),
        RigTaaRecordAction::Publish(_)
    ));
}

#[test]
fn destination_replacement_resets_continuity() {
    let source = BufferId::new();
    let destination = BufferId::new();
    let alloc = allocations();
    let mut state = published(source, destination, &alloc, 0);

    let replaced = BufferId::new();
    let third = epoch(3, source, replaced, &alloc, 0);
    assert_eq!(
        copy_frame(&mut state, true, Some(&third)),
        RigTaaCopyAction::Invalidate,
        "a recycled destination cannot inherit the old metadata"
    );
    assert_eq!(state.destination, Some(replaced));
    assert!(state.previous.is_none());
    assert!(state.header_valid, "the clear is owed until it is queued");
    ack_clear_now(&mut state);
    assert!(!state.header_valid);
    assert_eq!(
        record_frame(&mut state, true, Some(&confirmed(&third))),
        RigTaaRecordAction::Idle
    );

    let fourth = epoch(4, source, replaced, &alloc, 0);
    assert!(matches!(
        copy_frame(&mut state, true, Some(&fourth)),
        RigTaaCopyAction::Submit(_)
    ));
    assert!(matches!(
        record_frame(&mut state, true, Some(&confirmed(&fourth))),
        RigTaaRecordAction::Publish(_)
    ));
}

#[test]
fn the_previous_watermark_governs_the_palette_prefix_and_slot_validity() {
    let source = BufferId::new();
    let destination = BufferId::new();
    let mut previous_alloc = allocations();
    previous_alloc[1] = slot(7, 0, 4);
    previous_alloc[2] = slot(8, 4, 4);
    let (mut state, _) = seeded(source, destination, &previous_alloc, 8);

    let mut current = previous_alloc.clone();
    current[2] = RigAllocation::default();
    let second = epoch(2, source, destination, &current, 8);
    let (copy, _) = run(&mut state, true, &second, &confirmed(&second));
    let RigTaaCopyAction::Submit(plan) = copy else {
        panic!("consecutive epoch: {copy:?}");
    };
    assert_eq!(plan.header[3], 8, "the previous epoch's watermark");
    assert_eq!(plan.slots[1], [1, 7, 0, 4]);
    assert_eq!(plan.slots[2], [0; 4]);
    let palette = plan
        .regions
        .iter()
        .find(|region| region.destination_offset == PALETTE_OFFSET)
        .expect("a live prefix is copied");
    assert_eq!(palette.bytes, 8 * BONE_BYTES);
}

#[test]
fn an_empty_palette_copies_the_tables_and_publishes() {
    let source = BufferId::new();
    let destination = BufferId::new();
    let alloc = allocations();
    let (mut state, _) = seeded(source, destination, &alloc, 0);
    let second = epoch(2, source, destination, &alloc, 0);
    let (copy, record) = run(&mut state, true, &second, &confirmed(&second));
    let RigTaaCopyAction::Submit(plan) = copy else {
        panic!("empty palette still rotates: {copy:?}");
    };
    assert!(plan
        .regions
        .iter()
        .all(|region| region.destination_offset != PALETTE_OFFSET));
    assert_eq!(plan.header[3], 0);
    assert!(matches!(record, RigTaaRecordAction::Publish(_)));
    assert!(state.header_valid);
}

/// A destination change always asks for a clear, even when no valid header is owed: a buffer this
/// service did not create may carry a stale snapshot from an earlier life.
#[test]
fn a_destination_change_always_requests_a_clear() {
    let source = BufferId::new();
    let alloc = allocations();
    let mut state = RigTaaSnapshotState::default();
    let first = BufferId::new();
    // First observation adopts the destination, so its own header is cleared before anything else.
    assert_eq!(
        copy_frame(&mut state, true, Some(&epoch(1, source, first, &alloc, 0))),
        RigTaaCopyAction::Invalidate
    );
    assert_eq!(state.destination, Some(first));
    assert!(
        state.header_valid,
        "a new buffer is presumed possibly-valid"
    );
    ack_clear_now(&mut state);
    assert!(!state.header_valid);

    // A different destination arrives while nothing is owed.
    let second = BufferId::new();
    assert_eq!(
        copy_frame(&mut state, true, Some(&epoch(2, source, second, &alloc, 0))),
        RigTaaCopyAction::Invalidate
    );
    assert!(
        state.header_valid,
        "a recycled destination is presumed possibly-valid until it is cleared"
    );
    ack_clear_now(&mut state);
    assert!(!state.header_valid);
}
// ---------------------------------------------------------------- duplicate-epoch identity

/// A duplicate frame number is only a duplicate when it names the same source, palette and
/// allocation state. A source change inside an already published epoch must invalidate, and the
/// replaced source must not inherit the old publication.
#[test]
fn a_source_change_in_the_published_epoch_is_not_a_duplicate() {
    let source = BufferId::new();
    let destination = BufferId::new();
    let alloc = allocations();
    let mut state = published(source, destination, &alloc, 0);

    let replaced_source = BufferId::new();
    let replaced = epoch(2, replaced_source, destination, &alloc, 0);
    assert_eq!(
        copy_frame(&mut state, true, Some(&replaced)),
        RigTaaCopyAction::Invalidate,
        "a same-frame source change must not leave the old valid header advertised"
    );
    // The clear is queued to the safe destination, so the obligation is discharged.
    ack_clear_now(&mut state);
    assert!(!state.header_valid);
    assert!(state.previous.is_none());
    assert!(state.copied.is_none());
    assert!(state.published.is_none());

    // Matching receipts for the replaced source do not restore the old publication: the epoch only
    // seeds, and no valid header is advertised.
    assert_eq!(
        record_frame(&mut state, true, Some(&confirmed(&replaced))),
        RigTaaRecordAction::Idle
    );
    assert!(!state.header_valid);
    assert_eq!(state.previous.as_ref().unwrap().source, replaced_source);

    // The next consecutive epoch copies from the replacement and publishes.
    let third = epoch(3, replaced_source, destination, &alloc, 0);
    let copy = copy_frame(&mut state, true, Some(&third));
    assert!(matches!(copy, RigTaaCopyAction::Submit(_)));
    let record = record_frame(&mut state, true, Some(&confirmed(&third)));
    assert!(matches!(record, RigTaaRecordAction::Publish(_)));
    acknowledge_queued_now(&mut state, &copy, &record);
    assert!(state.header_valid);
}

/// The same holds when the copy is still pending publication: the submitted rotation no longer
/// describes the epoch and must not be published.
#[test]
fn a_source_change_during_a_pending_copy_is_not_a_duplicate() {
    let source = BufferId::new();
    let destination = BufferId::new();
    let alloc = allocations();
    let mut state = published(source, destination, &alloc, 0);

    let third = epoch(3, source, destination, &alloc, 0);
    assert!(matches!(
        copy_frame(&mut state, true, Some(&third)),
        RigTaaCopyAction::Submit(_)
    ));
    let replaced_source = BufferId::new();
    let replaced = epoch(3, replaced_source, destination, &alloc, 0);
    assert_eq!(
        copy_frame(&mut state, true, Some(&replaced)),
        RigTaaCopyAction::Invalidate,
        "the submitted copy no longer describes this epoch"
    );
    ack_clear_now(&mut state);
    assert!(!state.header_valid);
    assert!(state.copied.is_none(), "the stale rotation is dropped");

    // The record stage confirms the replaced source's receipts but publishes nothing for the
    // invalidated copy.
    assert_eq!(
        record_frame(&mut state, true, Some(&confirmed(&replaced))),
        RigTaaRecordAction::Idle
    );
    assert!(!state.header_valid);
    assert_eq!(state.previous.as_ref().unwrap().source, replaced_source);
}

/// Published slot records describe one allocation state; a duplicate with changed identities,
/// ranges or watermark cannot keep them readable.
#[test]
fn changed_allocation_state_in_the_published_epoch_is_not_a_duplicate() {
    let source = BufferId::new();
    let destination = BufferId::new();
    let mut previous = allocations();
    previous[1] = slot(7, 0, 4);
    let mut state = published(source, destination, &previous, 4);

    let mut reused = previous.clone();
    reused[1] = slot(8, 0, 4);
    assert_eq!(
        copy_frame(
            &mut state,
            true,
            Some(&epoch(2, source, destination, &reused, 4))
        ),
        RigTaaCopyAction::Invalidate,
        "a reused allocation identity"
    );
    ack_clear_now(&mut state);
    assert!(
        !state.header_valid,
        "the obsolete slot records must not stay readable"
    );
    assert!(state.published.is_none());

    // Re-publish the original identity, then each remaining change invalidates on its own.
    state = published(source, destination, &previous, 4);
    let mut resized = previous.clone();
    resized[1] = slot(7, 0, 5);
    assert_eq!(
        copy_frame(
            &mut state,
            true,
            Some(&epoch(2, source, destination, &resized, 4))
        ),
        RigTaaCopyAction::Invalidate,
        "a changed bone count"
    );
    ack_clear_now(&mut state);

    state = published(source, destination, &previous, 4);
    assert_eq!(
        copy_frame(
            &mut state,
            true,
            Some(&epoch(2, source, destination, &previous, 8))
        ),
        RigTaaCopyAction::Invalidate,
        "a changed watermark changes the copied palette prefix"
    );
    ack_clear_now(&mut state);
    assert!(!state.header_valid);
}

/// A clear that could not be queued is an obligation, not a completed invalidation: losing the
/// destination must not let the same buffer return with its old valid header.
#[test]
fn a_clear_without_a_target_keeps_the_obligation() {
    let source = BufferId::new();
    let destination = BufferId::new();
    let alloc = allocations();
    let mut state = published(source, destination, &alloc, 0);
    assert!(state.header_valid);

    // The destination resource is gone, so `current_epoch_view` cannot be built and neither stage
    // has anywhere to queue its clear.
    let (copy, record) = run_without_clear_target(&mut state, true, None, None);
    assert_eq!(copy, RigTaaCopyAction::Invalidate);
    assert_eq!(record, RigTaaRecordAction::Invalidate);
    assert!(
        state.header_valid,
        "an invalidation that could not be queued must stay owed"
    );

    // The same buffer returns within the same epoch: the obligation is still owed, so the stage
    // must ask for the clear before anything may reuse the destination.
    let resumed = epoch(2, source, destination, &alloc, 0);
    assert_eq!(
        copy_frame(&mut state, true, Some(&resumed)),
        RigTaaCopyAction::Invalidate
    );
    assert!(state.header_valid);
    assert!(
        state.copied.is_none(),
        "nothing may be reused before the clear"
    );

    // A safe destination is back, so the queueable clear discharges the obligation and the epoch
    // re-seeds.
    ack_clear_now(&mut state);
    assert!(!state.header_valid);
    assert_eq!(
        copy_frame(&mut state, true, Some(&resumed)),
        RigTaaCopyAction::Idle,
        "the restored epoch seeds instead of copying"
    );
    assert_eq!(
        record_frame(&mut state, true, Some(&confirmed(&resumed))),
        RigTaaRecordAction::Idle
    );
    assert!(!state.header_valid, "a seed frame publishes nothing");
    assert_eq!(state.previous.as_ref().unwrap().frame, 2);

    // The next consecutive epoch publishes again.
    let third = epoch(3, source, destination, &alloc, 0);
    let copy = copy_frame(&mut state, true, Some(&third));
    assert!(matches!(copy, RigTaaCopyAction::Submit(_)));
    let record = record_frame(&mut state, true, Some(&confirmed(&third)));
    assert!(matches!(record, RigTaaRecordAction::Publish(_)));
    acknowledge_queued_now(&mut state, &copy, &record);
    assert!(state.header_valid);
}

/// Restoration in a *later* epoch must clear the pending invalidation just the same: the clear is
/// owed against the buffer, not against the epoch it was lost in.
#[test]
fn a_late_restoration_still_clears_the_pending_invalidation_first() {
    let source = BufferId::new();
    let destination = BufferId::new();
    let alloc = allocations();
    let mut state = published(source, destination, &alloc, 0);

    // Frames 3..6 never see the destination.
    for frame in 3..=6 {
        let (copy, record) = run_without_clear_target(&mut state, true, None, None);
        assert_eq!(copy, RigTaaCopyAction::Invalidate, "frame {frame}");
        assert_eq!(record, RigTaaRecordAction::Invalidate, "frame {frame}");
        assert!(state.header_valid, "frame {frame}: the clear is still owed");
    }

    // The same buffer returns at frame 7: the clear comes first, then the epoch seeds.
    let seventh = epoch(7, source, destination, &alloc, 0);
    assert_eq!(
        copy_frame(&mut state, true, Some(&seventh)),
        RigTaaCopyAction::Invalidate
    );
    ack_clear_now(&mut state);
    assert!(!state.header_valid);
    assert_eq!(
        copy_frame(&mut state, true, Some(&seventh)),
        RigTaaCopyAction::Idle
    );
    assert_eq!(
        record_frame(&mut state, true, Some(&confirmed(&seventh))),
        RigTaaRecordAction::Idle
    );
    assert_eq!(state.previous.as_ref().unwrap().frame, 7);
}

/// An invalidation *request* is not a queued write: both sinks keep the obligation until a safe
/// destination actually accepted the clear.
#[test]
fn a_requested_invalidation_is_not_a_queued_one() {
    let source = BufferId::new();
    let destination = BufferId::new();
    let alloc = allocations();

    // Copy-stage sink: the destination is gone, so its request cannot be queued.
    let mut state = published(source, destination, &alloc, 0);
    let (copy, record) = run_without_clear_target(&mut state, true, None, None);
    assert_eq!(copy, RigTaaCopyAction::Invalidate);
    assert_eq!(record, RigTaaRecordAction::Invalidate);
    assert!(state.header_valid, "requested, not queued");
    ack_clear_now(&mut state);
    assert!(!state.header_valid, "queued, then discharged");

    // Record-stage sink: the copy rotated, then the receipts failed. The copy stage asked for no
    // clear, and without a destination the record's request stays owed too.
    let mut state = published(source, destination, &alloc, 0);
    let third = epoch(3, source, destination, &alloc, 0);
    assert!(matches!(
        copy_frame(&mut state, true, Some(&third)),
        RigTaaCopyAction::Submit(_)
    ));
    let mut stale = confirmed(&third);
    stale.light = Some((1, third.source));
    let record = record_frame(&mut state, true, Some(&stale));
    assert_eq!(record, RigTaaRecordAction::Invalidate);
    assert!(
        state.header_valid,
        "the record's clear is owed until a safe target accepts it"
    );
    ack_clear_now(&mut state);
    assert!(
        !state.header_valid,
        "queued to a safe target, then discharged"
    );
}

// ---------------------------------------------------------------- failure-path write targets

/// Every GPU write needs a suitable target, including a failure-path clear: the buffer must be
/// at least one header long, writable as a copy destination, and never the live source.
#[test]
fn a_failure_path_clear_needs_a_safe_target() {
    let source = RigTaaBufferFacts {
        id: BufferId::new(),
        size: PREVIOUS_RIG_BYTES,
        copy_src: true,
        copy_dst: false,
    };
    let destination = RigTaaBufferFacts {
        id: BufferId::new(),
        size: PREVIOUS_RIG_BYTES,
        copy_src: false,
        copy_dst: true,
    };
    assert!(invalidation_target_usable(Some(source), destination));

    // `COPY_DST` missing: a write is not allowed at all.
    assert!(!invalidation_target_usable(
        Some(source),
        RigTaaBufferFacts {
            copy_dst: false,
            ..destination
        }
    ));
    // Fewer than one header of writable bytes.
    assert!(!invalidation_target_usable(
        Some(source),
        RigTaaBufferFacts {
            size: HEADER_BYTES - 1,
            ..destination
        }
    ));
    assert!(invalidation_target_usable(
        Some(source),
        RigTaaBufferFacts {
            size: HEADER_BYTES,
            ..destination
        }
    ));
    // Aliasing the live source: equal ids with every other condition satisfied, so aliasing is
    // the only reason to refuse.
    let alias = RigTaaBufferFacts {
        id: source.id,
        size: PREVIOUS_RIG_BYTES,
        copy_src: true,
        copy_dst: true,
    };
    assert!(
        !invalidation_target_usable(Some(alias), alias),
        "a clear must never be written into the live source"
    );
    assert!(
        !invalidation_target_usable(
            Some(source),
            RigTaaBufferFacts {
                id: source.id,
                ..destination
            }
        ),
        "a destination that aliases the source is refused despite COPY_DST"
    );
    // With no live source there is nothing to corrupt: a suitable destination is still writable.
    assert!(invalidation_target_usable(None, destination));
    assert!(!invalidation_target_usable(
        None,
        RigTaaBufferFacts {
            copy_dst: false,
            ..destination
        }
    ));
}

// ---------------------------------------------------------------- destination identity

/// A clear to another destination must never stand in for this one's. A snapshot valid on A, an
/// Off frame with B bound (whose clear the service queues), then A returning: A's own header was
/// never written, so A must still be cleared before it is reused.
#[test]
fn a_clear_to_another_destination_does_not_clear_a_returning_one() {
    let source = BufferId::new();
    let alloc = allocations();
    let a = BufferId::new();
    let b = BufferId::new();
    let mut state = published(source, a, &alloc, 0);
    assert!(state.header_valid);

    // Off with B bound: the frame observes B, so the clear goes to B and only B is acknowledged.
    let b_epoch = epoch(3, source, b, &alloc, 0);
    assert_eq!(
        copy_stage(&mut state, Some(b), false, Some(&b_epoch)),
        RigTaaCopyAction::Invalidate
    );
    state.clear_queued(b);
    assert!(!state.header_valid);
    assert_eq!(state.destination, Some(b));

    // A returns within the same epoch with demand on: its own header was never written, so
    // adopting A must ask for A's clear.
    let resumed = epoch(3, source, a, &alloc, 0);
    assert_eq!(
        copy_stage(&mut state, Some(a), true, Some(&resumed)),
        RigTaaCopyAction::Invalidate,
        "a clear written to another buffer cannot clear A"
    );
    state.clear_queued(a);
    assert!(!state.header_valid);

    // Then A seeds, and the next consecutive epoch publishes.
    assert_eq!(
        copy_frame(&mut state, true, Some(&resumed)),
        RigTaaCopyAction::Idle
    );
    assert_eq!(
        record_frame(&mut state, true, Some(&confirmed(&resumed))),
        RigTaaRecordAction::Idle
    );
    assert_eq!(state.previous.as_ref().unwrap().frame, 3);
    let fourth = epoch(4, source, a, &alloc, 0);
    let copy = copy_frame(&mut state, true, Some(&fourth));
    assert!(matches!(copy, RigTaaCopyAction::Submit(_)));
    let record = record_frame(&mut state, true, Some(&confirmed(&fourth)));
    assert!(matches!(record, RigTaaRecordAction::Publish(_)));
    acknowledge_queued(&mut state, Some(a), &copy, &record);
    assert!(state.header_valid);
}

/// The same, with no complete epoch view at all: the destination is still known from the binding,
/// so adopting it must ask for its own clear.
#[test]
fn a_clear_to_another_destination_does_not_clear_a_returning_one_without_an_epoch_view() {
    let source = BufferId::new();
    let alloc = allocations();
    let a = BufferId::new();
    let b = BufferId::new();
    let mut state = published(source, a, &alloc, 0);

    // The inputs are missing, so no epoch view can be built; B is nevertheless bound.
    assert_eq!(
        copy_frame_without_epoch(&mut state, Some(b), true),
        RigTaaCopyAction::Invalidate
    );
    state.clear_queued(b);
    assert!(!state.header_valid);
    assert_eq!(state.destination, Some(b));

    // A is bound again while the inputs are still missing: A must be cleared on adoption.
    assert_eq!(
        copy_frame_without_epoch(&mut state, Some(a), true),
        RigTaaCopyAction::Invalidate,
        "adopting A without an epoch view still needs its clear"
    );
    state.clear_queued(a);
    assert!(!state.header_valid);

    // Now that A is clear, neither stage asks again.
    assert_eq!(
        copy_frame_without_epoch(&mut state, Some(a), true),
        RigTaaCopyAction::Idle
    );
    assert_eq!(
        record_frame_without_epoch(&mut state, Some(a), true),
        RigTaaRecordAction::Idle
    );
}

/// B can take the 16-byte header clear but cannot hold a snapshot at all. Clearing B must not erase
/// the obligation that A owes on its own return.
#[test]
fn a_header_writable_destination_that_cannot_hold_a_snapshot_leaves_the_returning_one_owed() {
    let source = BufferId::new();
    let alloc = allocations();
    let a = BufferId::new();
    let b = BufferId::new();
    let source_facts = RigTaaBufferFacts {
        id: source,
        size: PREVIOUS_RIG_BYTES,
        copy_src: true,
        copy_dst: false,
    };
    let b_facts = RigTaaBufferFacts {
        id: b,
        size: HEADER_BYTES,
        copy_src: false,
        copy_dst: true,
    };
    assert!(
        invalidation_target_usable(Some(source_facts), b_facts),
        "B is header-writable, so its own clear is allowed"
    );
    assert!(
        !snapshot_buffers_usable(source_facts, b_facts, 0),
        "B cannot hold a full snapshot"
    );

    let mut state = published(source, a, &alloc, 0);
    // B is bound with no usable snapshot inputs.
    assert_eq!(
        copy_frame_without_epoch(&mut state, Some(b), true),
        RigTaaCopyAction::Invalidate
    );
    state.clear_queued(b);
    assert!(!state.header_valid, "B's own clear is discharged");

    // A returns with a full epoch view: A's validity is a separate obligation.
    let resumed = epoch(3, source, a, &alloc, 0);
    assert_eq!(
        copy_frame(&mut state, true, Some(&resumed)),
        RigTaaCopyAction::Invalidate,
        "clearing B says nothing about A"
    );
    state.clear_queued(a);
    assert!(!state.header_valid);
}

/// Restoration in a later epoch is cleared just the same.
#[test]
fn a_late_restoration_after_another_destination_is_still_cleared() {
    let source = BufferId::new();
    let alloc = allocations();
    let a = BufferId::new();
    let b = BufferId::new();
    let mut state = published(source, a, &alloc, 0);

    // Off with B bound at frame 3, then several Off frames with B still bound.
    assert_eq!(
        copy_stage(
            &mut state,
            Some(b),
            false,
            Some(&epoch(3, source, b, &alloc, 0))
        ),
        RigTaaCopyAction::Invalidate
    );
    state.clear_queued(b);
    for frame in 4..=6 {
        assert_eq!(
            copy_stage(
                &mut state,
                Some(b),
                false,
                Some(&epoch(frame, source, b, &alloc, 0))
            ),
            RigTaaCopyAction::Idle
        );
    }

    // A returns at frame 7: its own clear comes before reuse.
    let seventh = epoch(7, source, a, &alloc, 0);
    assert_eq!(
        copy_stage(&mut state, Some(a), true, Some(&seventh)),
        RigTaaCopyAction::Invalidate
    );
    state.clear_queued(a);
    assert!(!state.header_valid);
    assert_eq!(
        copy_frame(&mut state, true, Some(&seventh)),
        RigTaaCopyAction::Idle
    );
    assert_eq!(
        record_frame(&mut state, true, Some(&confirmed(&seventh))),
        RigTaaRecordAction::Idle
    );
    assert_eq!(state.previous.as_ref().unwrap().frame, 7);
}

/// An acknowledgement describes exactly the buffer its write went to: a clear or a publication
/// reported against another destination changes nothing.
#[test]
fn an_acknowledgement_only_describes_its_own_target() {
    let source = BufferId::new();
    let alloc = allocations();
    let a = BufferId::new();
    let b = BufferId::new();
    let mut state = published(source, a, &alloc, 0);
    assert!(state.header_valid);

    state.clear_queued(b);
    assert!(state.header_valid, "B's clear says nothing about A");
    state.clear_queued(a);
    assert!(!state.header_valid);

    state.metadata_queued(b);
    assert!(!state.header_valid, "B's publication says nothing about A");
    state.metadata_queued(a);
    assert!(state.header_valid);
}

/// The record stage has its own retirement and adoption paths, and both acknowledge their own
/// destination only.
#[test]
fn the_record_stage_acknowledges_only_its_own_destination() {
    let source = BufferId::new();
    let alloc = allocations();
    let a = BufferId::new();
    let b = BufferId::new();

    // A pending copy, then a stale receipt: the record retires and owes A's clear.
    let mut state = published(source, a, &alloc, 0);
    let third = epoch(3, source, a, &alloc, 0);
    assert!(matches!(
        copy_frame(&mut state, true, Some(&third)),
        RigTaaCopyAction::Submit(_)
    ));
    let mut stale = confirmed(&third);
    stale.light = Some((1, stale.epoch.source));
    assert_eq!(
        record_frame(&mut state, true, Some(&stale)),
        RigTaaRecordAction::Invalidate
    );
    state.clear_queued(b);
    assert!(state.header_valid, "the write went to A, not B");
    state.clear_queued(a);
    assert!(!state.header_valid);

    // A record alone adopts a new destination and owes its clear.
    let mut state = published(source, a, &alloc, 0);
    let for_b = confirmed(&epoch(4, source, b, &alloc, 0));
    assert_eq!(
        record_stage(&mut state, Some(b), true, Some(&for_b)),
        RigTaaRecordAction::Invalidate,
        "the record stage adopts B and owes B's clear"
    );
    assert_eq!(state.destination, Some(b));
    assert!(state.header_valid);
    state.clear_queued(b);
    assert!(!state.header_valid);
}

/// Once the adopted destination is clear, stable Off frames issue no copy, submission or clear.
#[test]
fn stable_off_after_an_adoption_clears_once() {
    let source = BufferId::new();
    let alloc = allocations();
    let a = BufferId::new();
    let b = BufferId::new();
    let mut state = published(source, a, &alloc, 0);

    assert_eq!(
        copy_stage(
            &mut state,
            Some(b),
            false,
            Some(&epoch(3, source, b, &alloc, 0))
        ),
        RigTaaCopyAction::Invalidate
    );
    state.clear_queued(b);
    assert!(!state.header_valid);

    for frame in 4..=8 {
        let later = epoch(frame, source, b, &alloc, 0);
        assert_eq!(
            copy_stage(&mut state, Some(b), false, Some(&later)),
            RigTaaCopyAction::Idle,
            "stable Off frame {frame} must not repeat a copy or a clear"
        );
        assert_eq!(
            record_stage(&mut state, Some(b), false, Some(&confirmed(&later))),
            RigTaaRecordAction::Idle
        );
    }
    assert!(!state.header_valid);
}
// ---------------------------------------------------------------- production schedule

/// The production `Render` schedule orders the real systems: the copy before BOTH uploaders and the
/// record after BOTH. The sets are unfolded, so a `.in_set` membership counts as the order the chain
/// gives it.
#[test]
fn the_render_schedule_orders_copy_before_both_uploaders_and_record_after_both() {
    use crate::lighting::{upload_light, LightingPlugin};

    let mut app = App::new();
    app.insert_sub_app(RenderApp, bevy::app::SubApp::new());
    app.add_plugins(LightingPlugin);
    crate::rig_palette::plugin(&mut app);

    assert!(
        render_runs_before(&mut app, copy_previous_rig_palette, upload_light),
        "the previous-data copy must submit before the light uploader queues its blob"
    );
    assert!(
        render_runs_before(
            &mut app,
            copy_previous_rig_palette,
            crate::rig_palette::upload_rig_palettes
        ),
        "the previous-data copy must submit before the palette uploader writes the tables"
    );
    assert!(
        render_runs_before(&mut app, upload_light, record_current_palette_epoch),
        "the record must confirm the light receipt after its uploader"
    );
    assert!(
        render_runs_before(
            &mut app,
            crate::rig_palette::upload_rig_palettes,
            record_current_palette_epoch
        ),
        "the record must confirm the palette receipt after its uploader"
    );
    // And the copy cannot be ordered after the record by accident.
    assert!(
        render_runs_before(
            &mut app,
            copy_previous_rig_palette,
            record_current_palette_epoch
        ),
        "the record belongs after the copy"
    );
    // Negative controls: the walker is not trivially true, and the chain has a real direction.
    assert!(
        !render_runs_before(
            &mut app,
            record_current_palette_epoch,
            copy_previous_rig_palette
        ),
        "the record must not precede the copy"
    );
    assert!(
        !render_runs_before(&mut app, record_current_palette_epoch, upload_light),
        "the record must not precede the light uploader"
    );
}

/// `a` runs before `b` in `app`'s `Render` schedule, by system identity.
///
/// A `.before`/`.after` against a system missing from the schedule is silently nothing, so identity
/// is compared through the constructed systems' `TypeId`; the dependency graph is walked with the
/// set hierarchy unfolded, so a set `a` sits in precedes what that set precedes.
fn render_runs_before<MA, MB>(
    app: &mut App,
    a: impl IntoSystem<(), (), MA>,
    b: impl IntoSystem<(), (), MB>,
) -> bool {
    use bevy::ecs::schedule::graph::Direction::{Incoming, Outgoing};
    use bevy::ecs::schedule::NodeId;
    use bevy::ecs::system::{IntoSystem, System};
    use std::any::TypeId;
    use std::collections::HashSet;

    let a = System::type_id(&IntoSystem::into_system(a));
    let b = System::type_id(&IntoSystem::into_system(b));
    let world = app
        .get_sub_app_mut(RenderApp)
        .expect("the test app installs a render sub-app")
        .world_mut();
    world.schedule_scope(Render, |world, schedule| {
        schedule
            .initialize(world)
            .expect("the Render schedule builds");
        let key = |id: TypeId| -> NodeId {
            schedule
                .systems()
                .expect("initialized")
                .find(|&(_, system)| System::type_id(&**system) == id)
                .map(|(key, _)| NodeId::System(key))
                .unwrap_or_else(|| panic!("{id:?} is not in the Render schedule"))
        };
        let (a, b) = (key(a), key(b));
        let dependency = schedule.graph().dependency().graph();
        let hierarchy = schedule.graph().hierarchy().graph();
        let (mut after, mut containers) = (HashSet::new(), HashSet::new());
        let mut work = vec![(a, false)];
        while let Some((node, is_after)) = work.pop() {
            let fresh = if is_after {
                after.insert(node)
            } else {
                containers.insert(node)
            };
            if !fresh {
                continue;
            }
            work.extend(
                dependency
                    .neighbors_directed(node, Outgoing)
                    .map(|next| (next, true)),
            );
            work.extend(
                hierarchy
                    .neighbors_directed(node, Incoming)
                    .map(|parent| (parent, false)),
            );
            if is_after {
                work.extend(
                    hierarchy
                        .neighbors_directed(node, Outgoing)
                        .map(|child| (child, true)),
                );
            }
        }
        after.contains(&b)
    })
}
