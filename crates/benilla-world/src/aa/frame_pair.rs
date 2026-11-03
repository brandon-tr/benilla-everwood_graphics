//! Two-slot per-frame payload retention shared by opaque depth, both mask lanes and, for the
//! private opt-in resolve, reconstructed color history.
//!
//! There is no GPU dependency here: `T` is a cached texture in production and a token in CPU
//! tests. Publication proves the owning pass was encoded, not that the GPU completed it or that the
//! reconstructed color is safe for consumer reuse; each consumer still validates its exact keys,
//! depth and masks.

use super::depth::DepthPublication;

pub(super) struct TemporalFramePair<T> {
    slots: [T; 2],
    publications: [DepthPublication; 2],
}

impl<T> TemporalFramePair<T> {
    pub(super) fn new(generation: u64, slots: [T; 2]) -> Self {
        Self {
            slots,
            publications: [
                DepthPublication::new(generation),
                DepthPublication::new(generation),
            ],
        }
    }

    /// Encode into the slot selected by `epoch` and publish only after the caller's pass succeeds.
    /// A new allocation never awaits prior current/previous content.
    pub(super) fn encode_current(
        &self,
        epoch: u64,
        generation: u64,
        encode: impl FnOnce(&T) -> bool,
    ) -> bool {
        if epoch == 0 {
            return false;
        }
        let slot = (epoch & 1) as usize;
        self.publications[slot].encode(epoch, generation, || encode(&self.slots[slot]))
    }

    /// The immediately preceding epoch's content for a resolve that is about to write the current
    /// slot. Unlike [`Self::previous`], this does not require current publication; the caller must
    /// still publish current only after its pass is encoded.
    pub(super) fn previous_for_write(&self, epoch: u64, generation: u64) -> Option<&T> {
        if epoch <= 1 {
            return None;
        }
        let previous = epoch - 1;
        let slot = (previous & 1) as usize;
        self.publications[slot]
            .is_current(previous, generation)
            .then_some(&self.slots[slot])
    }

    /// Strict current content for exactly this epoch and generation.
    pub(super) fn current(&self, epoch: u64, generation: u64) -> Option<&T> {
        if epoch == 0 {
            return None;
        }
        let slot = (epoch & 1) as usize;
        self.publications[slot]
            .is_current(epoch, generation)
            .then_some(&self.slots[slot])
    }

    /// The immediately preceding epoch's content, only while current is also published.
    /// A gap or an older surviving slot is never substituted for exact `epoch - 1`.
    pub(super) fn previous(&self, epoch: u64, generation: u64) -> Option<&T> {
        self.current(epoch, generation)?;
        let previous = epoch.checked_sub(1)?;
        if previous == 0 {
            return None;
        }
        let slot = (previous & 1) as usize;
        self.publications[slot]
            .is_current(previous, generation)
            .then_some(&self.slots[slot])
    }
}

#[cfg(test)]
mod tests;
