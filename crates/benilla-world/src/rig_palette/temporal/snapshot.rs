//! CPU planning, render-world state and the GPU copy that publish the previous rig snapshot.
//!
//! `wow_model.wgsl` skins from the live palette in the shared light buffer. Native TAA needs the
//! *previous* frame's skinning data as well, in a differently arranged buffer that the material
//! binds as `RigTaaBinds::previous`. This module makes that buffer hold a real, receipt-backed
//! previous epoch:
//!
//! 1. [`demand`] names the frame's eligible world views (a native-TAA mount on the world camera,
//!    compatible prepass, MSAA off). One world-scope demand covers every eligible view; a saved
//!    preference or a prepared attachment is not demand.
//! 2. [`copy_previous_rig_palette`] (set [`RigTaaSnapshotSet::CopyPrevious`]) submits one
//!    `copy_buffer_to_buffer` command buffer that reads the shared light buffer's *current* contents
//!    — the previous epoch's data, because this frame's uploads have not been queued yet — into
//!    the previous buffer, then submits it immediately.
//! 3. The production uploaders (`upload_light`, `upload_rig_palettes`, set
//!    [`RigTaaSnapshotSet::UploadCurrent`]) queue this frame's `write_buffer` transfers.
//! 4. [`record_current_palette_epoch`] (set [`RigTaaSnapshotSet::RecordCurrent`]) confirms BOTH
//!    upload receipts match this epoch's frame and source before retaining the epoch and writing the
//!    metadata that makes the copy readable.
//!
//! # Why the submission is immediate
//!
//! `wgpu::Queue::write_buffer` does not submit: its transfer runs at the start of the *next*
//! `Queue::submit`, before that submission's commands (`wgpu-27.0.1`, `api/queue.rs`). Encoding the
//! copy and letting Bevy's graph runner submit it after `PrepareResources` would therefore run the
//! copy *after* this frame's palette and light writes landed, copying current bytes under a
//! previous-frame label. So the copy owns a submit of its own inside `CopyPrevious`, and the
//! explicit set chain puts every writer of a copied region after it. The copied regions are the
//! frame block's trailing `MonkeyFrame`, the slot base table, the rig origin table and the live
//! palette prefix; the two uploaders above are the only production writers of those bytes
//! (`upload_prop_probes` writes the preceding probe region). No GPU wait is issued.
//!
//! # Validity
//!
//! A snapshot is valid only when a copy of a receipt-backed epoch was submitted and the current
//! epoch's own uploads were then confirmed by both receipts. The header alone is the consumer's
//! gate, so every failure — a gap, a source or destination change, a lost resource, an unusable
//! size or usage, aliasing, a missing or mismatched receipt — ends with an invalid header; stale
//! payload behind it stays unreadable. A duplicate frame number is only the same epoch when its
//! [identity](RigTaaEpochIdentity) (source, allocations, ranges and watermark) also matches, so a
//! replaced source or palette inside an already committed frame cannot inherit the old validity.
//!
//! Invalidation is tracked as an *obligation scoped to the destination actually bound*, not a
//! decision and not a global flag. The frame's binding is observed on every call — before demand
//! and before the epoch view, which may not exist — so the state never believes a buffer it is no
//! longer bound to is the destination. Adopting a different destination forgets the old claims and
//! always asks for that buffer's own clear, and the acknowledgement takes the identity of the
//! buffer the write went to: a clear written to B can never stand in for A's. A destination that
//! disappears therefore keeps its obligation, the same buffer returning — in the same epoch or
//! later — is cleared before anything may reuse it, and a clear that cannot be queued never counts
//! as done. Failure-path clears validate their target like every other GPU write: a buffer without
//! `COPY_DST`, shorter than one header, or the live source itself is never written; the obligation
//! waits for a safe destination instead. No demand retires the retained epoch and, once a safe
//! target exists, invalidates a formerly valid header exactly once.
//!
//! # Boundaries
//!
//! A submitted copy is not GPU completion, not TAA execution and not per-view colour history: this
//! module neither reads the buffer back nor schedules a reconstruction, and camera cuts, teleports
//! and target-history management stay with the later native-TAA lifecycle. Nothing here enables
//! native TAA or mounts it on a production camera.

use std::sync::Arc;

use bevy::ecs::schedule::SystemSet;
use bevy::prelude::*;
use bevy::render::extract_resource::ExtractResourcePlugin;
use bevy::render::render_resource::*;
use bevy::render::renderer::{RenderDevice, RenderQueue};
use bevy::render::{Render, RenderApp, RenderSystems};

use super::*;
use crate::lighting::SharedLightBuffer;
use crate::mesh_tag::MAX_RIG_SLOTS;
use crate::rig_palette::{
    rig_publication_complete, RigAllocation, RigPaletteExtract, MAX_PALETTE_BONES,
};
use benilla_assets::materials::RigTaaBinds;

pub(crate) mod demand;

pub(crate) use demand::{extract_rig_taa_world_views, RigTaaWorldDemand};

/// What the last upload of each source put on the GPU, as `(frame, source)`.
///
/// Written by the production uploaders and read by [`record_current_palette_epoch`]; a missing half
/// means that upload did not land this frame.
#[derive(Resource, Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct RigTaaUploadReceipts {
    pub(crate) light: Option<(u32, BufferId)>,
    pub(crate) palette: Option<(u32, BufferId)>,
}

/// An uploaded epoch whose content is known to be on the GPU: the copy's previous side and the
/// only thing a later epoch may label "previous".
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct RigTaaUploadedEpoch {
    pub(crate) frame: u32,
    pub(crate) source: BufferId,
    pub(crate) allocations: Arc<Vec<RigAllocation>>,
    pub(crate) watermark: u32,
}

/// One buffer-to-buffer range of the previous-data copy.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct RigTaaCopy {
    pub(crate) source_offset: u64,
    pub(crate) destination_offset: u64,
    pub(crate) bytes: u64,
}

/// The exact GPU work one epoch's previous-copy needs, plus the metadata that describes it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct RigTaaCopyPlan {
    pub(crate) header: [u32; 4],
    pub(crate) slots: Vec<[u32; 4]>,
    pub(crate) regions: Vec<RigTaaCopy>,
}

impl RigTaaCopyPlan {
    /// The highest source byte the plan reads (exclusive).
    pub(crate) fn source_extent(&self) -> u64 {
        self.regions
            .iter()
            .map(|region| region.source_offset + region.bytes)
            .max()
            .unwrap_or(0)
    }

    /// The highest destination byte the plan writes (exclusive).
    pub(crate) fn destination_extent(&self) -> u64 {
        self.regions
            .iter()
            .map(|region| region.destination_offset + region.bytes)
            .max()
            .unwrap_or(0)
    }
}

/// Plan the copy that turns the receipt-backed [`RigTaaUploadedEpoch`] and this epoch's published
/// allocation state into the previous buffer's contents.
///
/// `None` whenever the two epochs cannot be one rotation: a gap (`previous + 1 != current`, with
/// `u32` wrap so frame zero is valid), a different source buffer, or a publication whose shape is
/// not the full slot table. The metadata ABI is fixed: header `[valid, previous_frame,
/// current_frame, previous_watermark]`, slot `[valid, previous_id_low32, previous_id_high32,
/// previous_bone_count]`, an ineligible slot all zero. A slot stays valid only while its full
/// 64-bit allocation identity matches, its bone count is positive and equal, its base is equal, and
/// its rows lie inside the palette prefix this epoch copies.
pub(crate) fn plan_previous_rig_copy(
    previous: &RigTaaUploadedEpoch,
    current_frame: u32,
    current_source: BufferId,
    current_allocations: &[RigAllocation],
) -> Option<RigTaaCopyPlan> {
    if previous.frame.wrapping_add(1) != current_frame
        || previous.source != current_source
        || current_allocations.len() != MAX_RIG_SLOTS
        || previous.allocations.len() != MAX_RIG_SLOTS
        || previous.watermark as usize > MAX_PALETTE_BONES
    {
        return None;
    }
    let mut slots = vec![[0u32; 4]; MAX_RIG_SLOTS];
    for slot in 1..MAX_RIG_SLOTS {
        let p = previous.allocations[slot];
        let c = current_allocations[slot];
        let Some(end) = p.base.checked_add(p.bones) else {
            continue;
        };
        if p.id != 0
            && p.id == c.id
            && p.bones > 0
            && p.bones == c.bones
            && p.base == c.base
            && end <= previous.watermark
            && end as usize <= MAX_PALETTE_BONES
        {
            slots[slot] = [1, p.id as u32, (p.id >> 32) as u32, p.bones];
        }
    }
    let monkey_source = crate::lighting::per_frame_blob_bytes().checked_sub(MONKEY_BYTES)?;
    let mut regions = vec![
        RigTaaCopy {
            source_offset: monkey_source,
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
    ];
    let palette_bytes = previous.watermark as u64 * BONE_BYTES;
    if palette_bytes > 0 {
        regions.push(RigTaaCopy {
            source_offset: crate::rig_palette::palette_region_offset(),
            destination_offset: PALETTE_OFFSET,
            bytes: palette_bytes,
        });
    }
    Some(RigTaaCopyPlan {
        header: [1, previous.frame, current_frame, previous.watermark],
        slots,
        regions,
    })
}

/// The source bytes one epoch's copy reads: the whole per-frame prefix (its trailing
/// `MonkeyFrame`), the two rig tables and the live palette prefix.
pub(crate) fn copied_source_extent(watermark: u32) -> u64 {
    let mut end = crate::lighting::per_frame_blob_bytes()
        .max(crate::rig_palette::rig_table_region_offset() + SLOT_BASES_BYTES)
        .max(crate::rig_palette::rig_origin_region_offset() + SLOT_ORIGINS_BYTES);
    let palette_bytes = watermark as u64 * BONE_BYTES;
    if palette_bytes > 0 {
        end = end.max(crate::rig_palette::palette_region_offset() + palette_bytes);
    }
    end
}

/// A buffer's identity, extent and copy capabilities, without a GPU handle.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct RigTaaBufferFacts {
    pub(crate) id: BufferId,
    pub(crate) size: u64,
    pub(crate) copy_src: bool,
    pub(crate) copy_dst: bool,
}

impl RigTaaBufferFacts {
    fn of(buffer: &Buffer) -> Self {
        Self {
            id: buffer.id(),
            size: buffer.size(),
            copy_src: buffer.usage().contains(BufferUsages::COPY_SRC),
            copy_dst: buffer.usage().contains(BufferUsages::COPY_DST),
        }
    }
}

/// Whether the two buffers can carry one snapshot: distinct storage (a copy cannot alias source and
/// destination), big enough for every region this watermark touches, and with the copy
/// capabilities the submission needs.
pub(crate) fn snapshot_buffers_usable(
    source: RigTaaBufferFacts,
    destination: RigTaaBufferFacts,
    watermark: u32,
) -> bool {
    source.id != destination.id
        && source.copy_src
        && source.size >= copied_source_extent(watermark)
        && destination.copy_dst
        && destination.size >= PREVIOUS_RIG_BYTES
}

/// One render frame's published view of the world rig data, before this frame's uploads are queued.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct RigTaaEpochView {
    pub(crate) frame: u32,
    pub(crate) source: BufferId,
    pub(crate) destination: BufferId,
    pub(crate) allocations: Arc<Vec<RigAllocation>>,
    pub(crate) watermark: u32,
}

/// Build the epoch view from the extracted publication and the two GPU buffers, or `None` when any
/// input is missing, incomplete, unusable or aliased.
fn current_epoch_view(
    published: Option<&RigPaletteExtract>,
    shared: Option<&SharedLightBuffer>,
    binds: Option<&RigTaaBinds>,
) -> Option<RigTaaEpochView> {
    let (published, shared, binds) = (published?, shared?, binds?);
    let frame = published.frame?;
    if !rig_publication_complete(published) {
        return None;
    }
    if !snapshot_buffers_usable(
        RigTaaBufferFacts::of(&shared.0),
        RigTaaBufferFacts::of(&binds.previous),
        published.watermark,
    ) {
        return None;
    }
    Some(RigTaaEpochView {
        frame,
        source: shared.0.id(),
        destination: binds.previous.id(),
        allocations: Arc::clone(&published.allocations),
        watermark: published.watermark,
    })
}

/// Everything one epoch's rotation and its slot metadata claim: the frame, the source the bytes
/// actually came from, and the allocation state the slot records were derived from.
///
/// A duplicate frame number is only the same epoch when this whole identity matches. A source
/// replacement, a freed/reused allocation, a changed range or a different watermark inside an
/// already committed frame names a *different* epoch and must not inherit the old validity.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct RigTaaEpochIdentity {
    frame: u32,
    source: BufferId,
    allocations: Arc<Vec<RigAllocation>>,
    watermark: u32,
}

impl RigTaaEpochIdentity {
    fn of(epoch: &RigTaaEpochView) -> Self {
        Self {
            frame: epoch.frame,
            source: epoch.source,
            allocations: Arc::clone(&epoch.allocations),
            watermark: epoch.watermark,
        }
    }
}

/// The retained epoch and the destination state around it.
///
/// `copied` records the epoch a copy was submitted for and `published` the epoch whose confirmed
/// metadata was queued; both are identities, so a duplicate frame number with different inputs is
/// detected rather than reused. Frame `0` after a `u32` wrap is a real value, never a sentinel.
///
/// `destination` is the buffer the frame actually has bound, observed every call — including when
/// demand is Off and when no complete epoch view can be built — so it never lingers on a buffer
/// that is no longer bound.
///
/// `header_valid` is an *obligation scoped to that observed destination*: it may still advertise a
/// valid snapshot that this service has not cleared. Adopting a different destination forgets the
/// old claims and presumes the newly adopted buffer may carry a stale header, so it always asks for
/// its own clear; a clear queued to B can never discharge A's obligation. [`Self::clear_queued`]
/// and [`Self::metadata_queued`] take the buffer the write actually went to and do nothing when it
/// is not the observed one, so an acknowledgement can only ever describe its own target.
#[derive(Resource, Debug, Default)]
pub(crate) struct RigTaaSnapshotState {
    previous: Option<RigTaaUploadedEpoch>,
    destination: Option<BufferId>,
    copied: Option<RigTaaEpochIdentity>,
    published: Option<RigTaaEpochIdentity>,
    header_valid: bool,
}

impl RigTaaSnapshotState {
    /// The system queued the confirmed metadata to `destination`.
    pub(crate) fn metadata_queued(&mut self, destination: BufferId) {
        if self.destination == Some(destination) {
            self.header_valid = true;
        }
    }

    /// The system queued the 16-byte clear to `destination`; only that buffer's obligation is
    /// discharged, because a clear written to another buffer says nothing about this one.
    pub(crate) fn clear_queued(&mut self, destination: BufferId) {
        if self.destination == Some(destination) {
            self.header_valid = false;
        }
    }

    /// Drop every claim about the epoch and the destination without discharging the header
    /// obligation. Used when a decision's effect could not be applied (nothing may describe work
    /// that was never submitted) and when a different destination is adopted.
    fn reset(&mut self) {
        self.previous = None;
        self.copied = None;
        self.published = None;
    }
}

/// What the copy stage must do this render frame.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum RigTaaCopyAction {
    /// Submit this plan's regions; the record stage writes the metadata once receipts confirm it.
    Submit(RigTaaCopyPlan),
    /// Write an invalid header so stale payload cannot be read.
    Invalidate,
    Idle,
}

/// What the record stage must do this render frame.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum RigTaaRecordAction {
    /// Write this epoch's confirmed header and slot records.
    Publish(RigTaaCopyPlan),
    /// Write an invalid header so a provisional or stale one cannot be read.
    Invalidate,
    Idle,
}

/// Adopt the destination this frame actually has bound, before any demand or epoch check.
///
/// `None` keeps the last observed identity: nothing is bound, so the same buffer returning must
/// still be recognised (and its owed clear, if any, still owed). A different buffer is an
/// adoption: the old epoch claims are dropped, and the new buffer is presumed able to carry a
/// stale header, so returning `true` makes the caller ask for its own clear first.
fn observe_destination(state: &mut RigTaaSnapshotState, observed: Option<BufferId>) -> bool {
    let Some(observed) = observed else {
        return false;
    };
    if state.destination == Some(observed) {
        return false;
    }
    state.reset();
    state.destination = Some(observed);
    state.header_valid = true;
    true
}

/// The copy stage: decide whether this epoch rotates the destination.
///
/// Called before any uploader queues a write. The frame's actual destination is observed first; a
/// newly adopted buffer is cleared before anything else, because its own header was never written.
/// Every other path that cannot produce a receipt-backed previous epoch retires the retained one
/// and, while the observed destination may still advertise a valid snapshot, asks for the one
/// clear that discharges that obligation.
pub(crate) fn copy_stage(
    state: &mut RigTaaSnapshotState,
    observed_destination: Option<BufferId>,
    demand: bool,
    current: Option<&RigTaaEpochView>,
) -> RigTaaCopyAction {
    // The destination is observed before demand and before the epoch view: a clear is owed per
    // *buffer*, and neither is relevant to whether a newly bound buffer may carry a stale header.
    if observe_destination(state, observed_destination) {
        return RigTaaCopyAction::Invalidate;
    }
    let Some(current) = current else {
        return retire(state);
    };
    if !demand {
        return retire(state);
    }

    // A duplicate is the same frame *and* the same source, palette and allocation state. Anything
    // else under the same frame number is a replaced epoch whose committed rotation/metadata no
    // longer describes the inputs.
    let identity = RigTaaEpochIdentity::of(current);
    let committed = state
        .copied
        .clone()
        .filter(|copied| copied.frame == current.frame)
        .or_else(|| {
            state
                .published
                .clone()
                .filter(|published| published.frame == current.frame)
        });
    if let Some(committed) = committed {
        if committed == identity {
            // This epoch already rotated (or already confirmed one). A second call must not copy
            // again.
            return RigTaaCopyAction::Idle;
        }
        return retire(state);
    }

    let Some(previous) = state.previous.as_ref() else {
        // First demand, or a re-seed after a gap: the destination stays invalid until a later
        // consecutive epoch has a receipt-backed previous side to copy.
        return retire(state);
    };
    match plan_previous_rig_copy(
        previous,
        current.frame,
        current.source,
        &current.allocations,
    ) {
        Some(plan) => {
            state.copied = Some(identity);
            RigTaaCopyAction::Submit(plan)
        }
        None => retire(state),
    }
}

/// The record stage's inputs: this epoch's view plus the two uploaders' receipts.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct RigTaaRecordInput {
    pub(crate) epoch: RigTaaEpochView,
    pub(crate) light: Option<(u32, BufferId)>,
    pub(crate) palette: Option<(u32, BufferId)>,
}

/// The record stage: confirm the epoch or invalidate the record.
///
/// The frame's actual destination is observed first, exactly as in the copy stage, so a buffer
/// adopted on a frame that never reaches the epoch logic still has its clear requested. The epoch
/// is retained only when BOTH receipts name this epoch's frame and source, which is what ties the
/// retained metadata to bytes that are actually on the GPU. The metadata write happens here, after
/// the copy, so a valid header is never exposed for an epoch whose upload has not been confirmed;
/// a seed epoch (no copy) records the epoch but publishes nothing.
pub(crate) fn record_stage(
    state: &mut RigTaaSnapshotState,
    observed_destination: Option<BufferId>,
    demand: bool,
    input: Option<&RigTaaRecordInput>,
) -> RigTaaRecordAction {
    if observe_destination(state, observed_destination) {
        return RigTaaRecordAction::Invalidate;
    }
    let Some(input) = input else {
        return retire_record(state);
    };
    if !demand {
        return retire_record(state);
    }
    if state.destination != Some(input.epoch.destination) {
        return retire_record(state);
    }
    let frame = input.epoch.frame;
    let source = input.epoch.source;
    if input.light != Some((frame, source)) || input.palette != Some((frame, source)) {
        // Missing, stale, mismatched or incomplete receipts: neither the retained epoch nor any
        // header it may have advertised survives.
        return retire_record(state);
    }
    let identity = RigTaaEpochIdentity::of(&input.epoch);
    if let Some(published) = state
        .published
        .as_ref()
        .filter(|published| published.frame == frame)
    {
        return if *published == identity {
            // A duplicate record for an epoch already confirmed, from the same inputs, changes
            // nothing.
            RigTaaRecordAction::Idle
        } else {
            // The same frame was already published from another source or allocation state: its
            // slot records describe inputs that no longer exist.
            retire_record(state)
        };
    }

    // A copy published this epoch's metadata only when the copy stage actually rotated for it, and
    // only when the copy still describes the same inputs. The plan is deterministic for the same
    // inputs, so the record recomputes it rather than retaining a second slot table.
    let copied = state.copied.clone().filter(|copied| copied.frame == frame);
    let plan = match &copied {
        Some(copied) if *copied != identity => {
            // The copy submitted for this epoch no longer matches its inputs: nothing may publish.
            return retire_record(state);
        }
        Some(_) => state.previous.as_ref().and_then(|previous| {
            plan_previous_rig_copy(previous, frame, source, &input.epoch.allocations)
        }),
        None => None,
    };
    if plan.is_none() && copied.is_some() {
        // A submitted copy that cannot be described any more must not stay advertised.
        state.reset();
        return invalidate_record(state);
    }

    state.previous = Some(RigTaaUploadedEpoch {
        frame,
        source,
        allocations: Arc::clone(&input.epoch.allocations),
        watermark: input.epoch.watermark,
    });
    match plan {
        Some(plan) => {
            state.published = Some(identity);
            RigTaaRecordAction::Publish(plan)
        }
        None => RigTaaRecordAction::Idle,
    }
}

/// Drop the retained epoch and every claim about the destination. The header obligation is not
/// discharged here: only the system can do that, after a clear has actually been queued.
fn retire(state: &mut RigTaaSnapshotState) -> RigTaaCopyAction {
    state.reset();
    invalidate_header(state)
}

/// Ask for a clear while the destination may still advertise a valid snapshot.
///
/// `header_valid` is the obligation, not a claim about a write: it survives an action that could
/// not be queued and is cleared only by [`RigTaaSnapshotState::clear_queued`].
fn invalidate_header(state: &mut RigTaaSnapshotState) -> RigTaaCopyAction {
    if state.header_valid {
        RigTaaCopyAction::Invalidate
    } else {
        RigTaaCopyAction::Idle
    }
}

fn retire_record(state: &mut RigTaaSnapshotState) -> RigTaaRecordAction {
    state.reset();
    invalidate_record(state)
}

fn invalidate_record(state: &mut RigTaaSnapshotState) -> RigTaaRecordAction {
    if state.header_valid {
        RigTaaRecordAction::Invalidate
    } else {
        RigTaaRecordAction::Idle
    }
}

/// The buffer a failure-path clear may target: at least one header long, writable as a copy
/// destination, and never the live source.
///
/// A failure path decides to clear without the full epoch being usable, so it cannot rely on
/// [`snapshot_buffers_usable`]'s whole-buffer checks; it still must not issue a GPU write to a
/// buffer that is too small, is not `COPY_DST`, or *is* the buffer the frame's data is read from.
/// When no safe target exists the caller keeps the invalidation obligation instead of writing.
pub(crate) fn invalidation_target_usable(
    source: Option<RigTaaBufferFacts>,
    destination: RigTaaBufferFacts,
) -> bool {
    destination.size >= HEADER_BYTES
        && destination.copy_dst
        && source.is_none_or(|source| source.id != destination.id)
}

/// Queue the 16-byte invalid header to the destination actually bound, returning that buffer's
/// identity so the acknowledgement can be tied to the write. `None` when no safe target exists,
/// which leaves the invalidation obligation owed.
fn clear_previous_header(
    queue: &RenderQueue,
    shared: Option<&SharedLightBuffer>,
    binds: Option<&RigTaaBinds>,
) -> Option<BufferId> {
    let binds = binds?;
    let destination = binds.previous.id();
    let source = shared.map(|shared| RigTaaBufferFacts::of(&shared.0));
    if !invalidation_target_usable(source, RigTaaBufferFacts::of(&binds.previous)) {
        return None;
    }
    invalidate_snapshot_header(queue, &binds.previous);
    Some(destination)
}

/// Submit the previous-data copy and return, without waiting on the GPU.
///
/// The submit is the ordering point: any `write_buffer` queued after this call is applied at a
/// later submit, so it cannot leak into these copies.
fn submit_previous_copy(
    device: &RenderDevice,
    queue: &RenderQueue,
    source: &Buffer,
    destination: &Buffer,
    plan: &RigTaaCopyPlan,
) {
    debug_assert!(plan.source_extent() <= source.size());
    debug_assert!(plan.destination_extent() <= destination.size());
    let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
        label: Some("rig_taa_previous_copy"),
    });
    for region in &plan.regions {
        encoder.copy_buffer_to_buffer(
            source,
            region.source_offset,
            destination,
            region.destination_offset,
            region.bytes,
        );
    }
    let _ = queue.submit(std::iter::once(encoder.finish()));
}

/// The metadata a consumer reads before the payload: header first, then the slot records.
///
/// The destination was validated as a whole-epoch target by [`current_epoch_view`] (full size,
/// `COPY_DST`, not the live source); these assertions pin that invariant at the write itself.
fn write_snapshot_metadata(queue: &RenderQueue, destination: &Buffer, plan: &RigTaaCopyPlan) {
    debug_assert!(
        destination.size() >= SLOT_META_OFFSET + SLOT_META_BYTES,
        "a metadata write needs room for header and slot table"
    );
    debug_assert!(
        destination.usage().contains(BufferUsages::COPY_DST),
        "a metadata write needs a copy destination"
    );
    queue.write_buffer(destination, 0, bytemuck::cast_slice(&plan.header));
    queue.write_buffer(
        destination,
        SLOT_META_OFFSET,
        bytemuck::cast_slice(&plan.slots),
    );
}

/// One 16-byte write is enough to make a snapshot unreadable: the header gates the payload.
fn invalidate_snapshot_header(queue: &RenderQueue, destination: &Buffer) {
    queue.write_buffer(destination, 0, &[0u8; HEADER_BYTES as usize]);
}

/// Render world (`PrepareResources`, [`RigTaaSnapshotSet::CopyPrevious`]): snapshot the previous
/// epoch's GPU contents before this frame's uploads are queued.
fn copy_previous_rig_palette(
    device: Res<RenderDevice>,
    queue: Res<RenderQueue>,
    demand: Res<RigTaaWorldDemand>,
    published: Option<Res<RigPaletteExtract>>,
    shared: Option<Res<SharedLightBuffer>>,
    binds: Option<Res<RigTaaBinds>>,
    mut state: ResMut<RigTaaSnapshotState>,
) {
    // The destination is read from the binding itself, so it is known even when the rest of the
    // epoch (source, publication, demand) is missing.
    let observed = binds.as_deref().map(|binds| binds.previous.id());
    let current = current_epoch_view(published.as_deref(), shared.as_deref(), binds.as_deref());
    match copy_stage(&mut state, observed, !demand.is_empty(), current.as_ref()) {
        RigTaaCopyAction::Submit(plan) => {
            let (Some(shared), Some(binds)) = (shared.as_deref(), binds.as_deref()) else {
                // The inputs vanished between the decision and its effect: no submission happened,
                // so nothing may describe one.
                state.reset();
                return;
            };
            submit_previous_copy(&device, &queue, &shared.0, &binds.previous, &plan);
        }
        RigTaaCopyAction::Invalidate => {
            // A clear is owed by whichever destination is bound now. Only a queued write to that
            // buffer discharges it; a missing or unsafe target leaves the obligation owed.
            if let Some(destination) =
                clear_previous_header(&queue, shared.as_deref(), binds.as_deref())
            {
                state.clear_queued(destination);
            }
        }
        RigTaaCopyAction::Idle => {}
    }
}

/// Render world (`PrepareResources`, [`RigTaaSnapshotSet::RecordCurrent`]): retain the epoch only
/// when both upload receipts confirm it, and publish the metadata that describes the copy.
fn record_current_palette_epoch(
    queue: Res<RenderQueue>,
    demand: Res<RigTaaWorldDemand>,
    published: Option<Res<RigPaletteExtract>>,
    shared: Option<Res<SharedLightBuffer>>,
    binds: Option<Res<RigTaaBinds>>,
    receipts: Res<RigTaaUploadReceipts>,
    mut state: ResMut<RigTaaSnapshotState>,
) {
    let observed = binds.as_deref().map(|binds| binds.previous.id());
    let current = current_epoch_view(published.as_deref(), shared.as_deref(), binds.as_deref());
    let input = current.map(|epoch| RigTaaRecordInput {
        epoch,
        light: receipts.light,
        palette: receipts.palette,
    });
    match record_stage(&mut state, observed, !demand.is_empty(), input.as_ref()) {
        RigTaaRecordAction::Publish(plan) => {
            let Some(binds) = binds.as_deref() else {
                // A publish without its destination cannot have been queued; nothing may claim it.
                state.reset();
                return;
            };
            let destination = binds.previous.id();
            write_snapshot_metadata(&queue, &binds.previous, &plan);
            state.metadata_queued(destination);
        }
        RigTaaRecordAction::Invalidate => {
            if let Some(destination) =
                clear_previous_header(&queue, shared.as_deref(), binds.as_deref())
            {
                state.clear_queued(destination);
            }
        }
        RigTaaRecordAction::Idle => {}
    }
}

#[derive(SystemSet, Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) enum RigTaaSnapshotSet {
    CopyPrevious,
    UploadCurrent,
    RecordCurrent,
}

/// Register the demand scan, its extraction, the snapshot state and the fixed `CopyPrevious ->
/// UploadCurrent -> RecordCurrent` order.
///
/// The uploaders join [`RigTaaSnapshotSet::UploadCurrent`] from their own modules; the chain here is
/// the only place that orders them against the copy and the record.
pub(crate) fn register(app: &mut App) {
    app.init_resource::<RigTaaWorldDemand>()
        .add_plugins(ExtractResourcePlugin::<RigTaaWorldDemand>::default())
        .add_plugins(ExtractResourcePlugin::<RigTaaBinds>::default())
        .add_systems(Last, extract_rig_taa_world_views);
    let Some(render) = app.get_sub_app_mut(RenderApp) else {
        return;
    };
    render
        .init_resource::<RigTaaUploadReceipts>()
        .init_resource::<RigTaaSnapshotState>()
        .init_resource::<RigTaaWorldDemand>()
        .configure_sets(
            Render,
            (
                RigTaaSnapshotSet::CopyPrevious,
                RigTaaSnapshotSet::UploadCurrent,
                RigTaaSnapshotSet::RecordCurrent,
            )
                .chain(),
        )
        .add_systems(
            Render,
            (
                copy_previous_rig_palette.in_set(RigTaaSnapshotSet::CopyPrevious),
                record_current_palette_epoch.in_set(RigTaaSnapshotSet::RecordCurrent),
            )
                .in_set(RenderSystems::PrepareResources),
        );
}

#[cfg(test)]
mod gpu_tests;
#[cfg(test)]
mod tests;
