//! Production wiring for native TAA: the resolve marker and the renderer-to-CPU history bridge.
//!
//! # The marker
//!
//! [`TemporalResolveView`] is inserted on every [`WorldCamera`] while the frame's *effective* mode is
//! [`AaEffective::Taa`] and removed otherwise. Auxiliary `Camera3d`s (portrait booths, reflection
//! probes) and the player-UI `Camera2d` never carry [`WorldCamera`], so they are never marked, never
//! jittered and never enter world history. A camera replacement and an effective-mode change both
//! converge on the same per-frame sync, so a stale marker cannot survive either.
//!
//! # Jitter
//!
//! The jitter schedule itself is app-owned (`benilla_app::aa_options`): it writes the stock
//! [`bevy::render::camera::TemporalJitter`] component on the world camera, so this crate and its GPU
//! fixtures keep their own explicit jitter control. The marker and the history bridge are
//! engine-owned because both derive from [`AaPolicy`] and [`TemporalHistory`] state; the app only
//! chooses the mode.
//!
//! # The history bridge
//!
//! The renderer owns the GPU history pair; this module owns the CPU [`TemporalHistory`] lifecycle.
//! A token is begun before extraction and settled only from evidence, never from elapsed time:
//!
//! - Extraction stamps each eligible view's attempt with the extraction epoch and the begun token's
//!   frame, and advances the bridge's `last_epoch`.
//! - The resolve marks an epoch committed only after its history write was encoded. A skipped node,
//!   a missing pipeline, an invalidated pair or a failed encode marks nothing.
//! - A begun frame is settled when a *later* extraction epoch exists. With pipelined rendering the
//!   render app for frame N is only returned to the main thread before extraction N + 1, so a later
//!   epoch proves render frame N completed. Under a synchronous renderer the same rule is trivially
//!   true. A frame whose render never attempted it is abandoned once extraction has moved past it.
//!
//! That ordering costs a begun token roughly every other frame under pipelining — the GPU history
//! still advances every rendered frame; only the CPU validity record lags. Committing before the
//! evidence exists would be a wall-clock race, and a stale commit is the one outcome the contract
//! forbids.

use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Mutex};

use bevy::prelude::*;

use super::resolve::TemporalResolveView;
use super::{AaEffective, AaPolicy, TemporalHistory};
use crate::view::WorldCamera;

/// The bridge's answer for a begun token.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Settle {
    /// The render frame that owns this token may still be running; keep the token in flight.
    Pending,
    /// The resolve encoded this frame's history write; commit the token.
    Committed,
    /// The render side moved past the frame without a successful resolve; abandon the token.
    Abandoned,
}

#[derive(Clone, Copy, Debug)]
struct Attempt {
    frame: u64,
    epoch: u64,
    generation: u64,
}

#[derive(Clone, Copy, Debug)]
struct Pending {
    frame: u64,
    /// The bridge's extraction epoch when the token was begun.
    since_epoch: u64,
}

#[derive(Default)]
struct ViewBridge {
    pending: Option<Pending>,
    /// Newest last; only attempts for frames that are not settled yet.
    attempts: VecDeque<Attempt>,
    committed_epoch: u64,
    committed_generation: u64,
}

#[derive(Default)]
struct BridgeState {
    /// The most recent extraction epoch, stamped by [`NativeResolveBridge::note_extraction`].
    last_epoch: u64,
    views: HashMap<Entity, ViewBridge>,
}

/// The only renderer-to-CPU channel for native history: an epoch-stamped publication shared by both
/// worlds behind one `Arc`. It says that commands were encoded, never that the GPU completed.
#[derive(Resource, Clone, Default)]
pub(super) struct NativeResolveBridge {
    state: Arc<Mutex<BridgeState>>,
}

impl NativeResolveBridge {
    fn lock(&self) -> std::sync::MutexGuard<'_, BridgeState> {
        self.state.lock().unwrap_or_else(|error| error.into_inner())
    }

    /// Record this extraction's epoch before its requests are read. The depth lane's epoch is used,
    /// so the render node, extraction and the settle rule all speak one counter.
    pub(super) fn note_extraction(&self, epoch: u64) {
        let mut state = self.lock();
        state.last_epoch = epoch;
    }

    /// The main world: a token was begun for `frame`. The next extraction stamps its attempt.
    pub(super) fn begin_frame(&self, view: Entity, frame: u64) {
        let mut state = self.lock();
        let since_epoch = state.last_epoch;
        let entry = state.views.entry(view).or_default();
        entry.pending = Some(Pending { frame, since_epoch });
    }

    /// The render world: this extraction is eligible for temporal work. Only the first extraction
    /// after a begun token records an attempt for it; a later extraction that still sees the token
    /// (its frame is not settled yet) must not invent a second, newer attempt.
    pub(super) fn attempt(&self, view: Entity, epoch: u64, generation: u64) {
        let mut state = self.lock();
        let Some(entry) = state.views.get_mut(&view) else {
            return;
        };
        let Some(pending) = entry.pending else {
            return;
        };
        if entry
            .attempts
            .iter()
            .any(|attempt| attempt.frame == pending.frame)
        {
            return;
        }
        entry.attempts.push_back(Attempt {
            frame: pending.frame,
            epoch,
            generation,
        });
        // Four frames of slack is already more than the settle rule needs; older attempts were
        // settled or belong to a replaced view.
        while entry.attempts.len() > 4 {
            entry.attempts.pop_front();
        }
    }

    /// The resolve: this epoch's history write was encoded for this exact generation. Anything else
    /// — a stale generation, a mismatched epoch, no attempt — publishes nothing.
    pub(super) fn mark_committed(&self, view: Entity, epoch: u64, generation: u64) {
        let mut state = self.lock();
        let Some(entry) = state.views.get_mut(&view) else {
            return;
        };
        let matches = entry
            .attempts
            .iter()
            .rev()
            .find(|attempt| attempt.epoch == epoch)
            .is_some_and(|attempt| attempt.generation == generation);
        if matches {
            entry.committed_epoch = epoch;
            entry.committed_generation = generation;
        }
    }

    /// Settle a begun token from the evidence published so far. See the module docs for why a later
    /// extraction epoch is the completion proof.
    pub(super) fn settle(&self, view: Entity, frame: u64) -> Settle {
        let mut state = self.lock();
        let last_epoch = state.last_epoch;
        let Some(entry) = state.views.get_mut(&view) else {
            return Settle::Abandoned;
        };
        let settled = if let Some(attempt) = entry
            .attempts
            .iter()
            .find(|attempt| attempt.frame == frame)
            .copied()
        {
            if last_epoch > attempt.epoch {
                Some(
                    if entry.committed_epoch == attempt.epoch
                        && entry.committed_generation == attempt.generation
                    {
                        Settle::Committed
                    } else {
                        Settle::Abandoned
                    },
                )
            } else {
                None
            }
        } else {
            match entry.pending {
                Some(pending) if pending.frame == frame => {
                    // Extraction has already happened since the token was begun and never stamped
                    // an attempt: the view was not eligible (multisampled, inactive, viewport or a
                    // missing target), so there is nothing to commit.
                    (last_epoch > pending.since_epoch).then_some(Settle::Abandoned)
                }
                // A different frame is pending: this token was superseded (an invalidation cleared
                // its in-flight state without this bridge hearing about it).
                Some(pending) if pending.frame > frame => Some(Settle::Abandoned),
                _ => None,
            }
        };
        if let Some(settled) = settled {
            entry.attempts.retain(|attempt| attempt.frame > frame);
            if entry.pending.is_some_and(|pending| pending.frame == frame) {
                entry.pending = None;
            }
            return settled;
        }
        Settle::Pending
    }

    /// Drop a replaced or removed view's bookkeeping.
    pub(super) fn forget(&self, view: Entity) {
        self.lock().views.remove(&view);
    }
}

/// Mark the world camera exactly while the effective mode is native TAA, and unmark it otherwise.
/// `Commands` are applied before extraction in the same frame.
pub(super) fn sync_native_resolve_marker(
    policy: Res<AaPolicy>,
    mut commands: Commands,
    cameras: Query<(Entity, Has<TemporalResolveView>), With<WorldCamera>>,
) {
    let want = policy.effective() == AaEffective::Taa;
    for (view, marked) in &cameras {
        if want && !marked {
            commands.entity(view).insert(TemporalResolveView);
        } else if !want && marked {
            commands.entity(view).remove::<TemporalResolveView>();
        }
    }
}

/// Begin and settle the CPU history from the renderer's own evidence. Runs after the policy
/// resolution and the marker sync, before extraction.
///
/// Only a camera that actually carries [`TemporalResolveView`] begins a token: a camera without the
/// marker cannot resolve, so beginning history for it would abandon on every attempt and churn the
/// history generation. The marker is inserted by commands that apply at the end of `Last`, so a
/// newly selected native camera begins on its second frame; settling an existing token still runs
/// for a view that just lost the marker.
pub(super) fn maintain_native_history(
    policy: Res<AaPolicy>,
    bridge: Res<NativeResolveBridge>,
    mut history: ResMut<TemporalHistory>,
    cameras: Query<(Entity, Has<TemporalResolveView>), With<WorldCamera>>,
) {
    for (view, marked) in &cameras {
        if let Some(token) = history.in_flight(view) {
            match bridge.settle(view, token.frame()) {
                // The render frame that owns the token may still be running. Nothing new may begin
                // while a token is in flight.
                Settle::Pending => continue,
                Settle::Committed => {
                    history.commit(token);
                }
                Settle::Abandoned => {
                    history.abandon(token);
                }
            }
        }
        if !marked || policy.effective() != AaEffective::Taa || !history.is_registered(view) {
            continue;
        }
        if let Some(token) = history.begin(view) {
            bridge.begin_frame(view, token.frame());
        }
    }
}

#[cfg(test)]
mod tests;
