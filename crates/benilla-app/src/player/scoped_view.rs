//! The spyglass zoom, `SPELL_AURA_FAR_SIGHT` (aura 76): a client-local camera override, unlike the
//! server's `SPELL_AURA_BIND_SIGHT` far sight ([`super::view_subject`]), so no packet carries it.
//! For the local player only (`0x5fa6d0`), the aura watcher `0x604d00` routes to `0x5ff350` (add)
//! and `0x612320` (remove), which match each effect's `EffectApplyAuraName` (`SpellRec+0x16c`)
//! against 76 and call `0x50d320(camera, EffectMiscValue[i])`: first person, locked (camera flag
//! `0x8` makes `SetCameraView` return early), with `[camera+0x40] = n × π/180`; removal restores
//! `π/2`, unlocks and selects view 1. The Ornate Spyglass (item 5507, spell 12883) passes 15, a 6×
//! zoom.
//!
//! [`ScopedView`] follows a change of the aura. A create's `PostInit` shows every standing aura
//! (`0x5fbb3c` → `0x5ff130`), which starts the scope, and our own arrival ends it at once and
//! selects view 1 (`0x5deb44`, `0x50d320(0)`); a fresh create runs no field watcher (`0x465d79`),
//! so a scope held across a worldport comes back off, under the same aura. A `/reload`'s new camera
//! holds no scope either, and the reload itself shows no aura. In the reference the display watcher
//! shows it again (`0x604990` → `0x60abe0` → `0x60ada6` → `0x5ff130` → `0x5ff350` → `0x5ff462`), so
//! a shapeshift or morph under a standing aura restarts the scope; that path is not built here.
//!
//! The ratio is applied to `[camera+0x40]`, which defaults to 90°; the world projection then turns
//! it into the vertical field of view by the frame's aspect (`0x5c3cc0`, [`benilla_world::view`]).

use bevy::prelude::*;

use crate::net::{ObjectStore, SelfPlayer};
use crate::ui_action::Spells;
use benilla_world::view::FovZoom;

/// `0x4c`, the aura name the reference's effect walk matches.
const SPELL_AURA_FAR_SIGHT: u32 = 76;

/// The reference's unzoomed `[camera+0x40]` in degrees (`π/2`, its constructor's value at
/// `0x50a706`), used only as a denominator.
const REFERENCE_DEFAULT_DEGREES: f32 = 90.0;

/// The held scope as a fraction of the normal field of view. `None` is no override;
/// `Some(1.0)` would still lock first person.
#[derive(Resource, Default)]
pub(crate) struct ScopedView {
    pub(crate) zoom: Option<f32>,
    /// This frame's arrival found the aura on our create: its `PostInit` started the scope and the
    /// arrival ends it (`0x5fbb3c`, `0x5deb44`), which selects view 1 from any view.
    pub(crate) arrival_end: bool,
    /// The aura the last look found (`0x604d00` compares the slots with their old values).
    aura: Option<u32>,
    /// A new camera was built since the last look.
    camera_rebuilt: bool,
}

impl ScopedView {
    /// While a scope is held the camera is locked in first person.
    pub(crate) fn active(&self) -> bool {
        self.zoom.is_some()
    }

    /// A new camera (`0x50c0e0`) holds no scope, and building it shows no aura.
    pub(super) fn new_camera(&mut self) {
        self.zoom = None;
        self.camera_rebuilt = true;
    }

    /// One look at the aura, `(spell, degrees)`, and whether our create arrived since the last.
    fn observe(&mut self, aura: Option<(u32, f32)>, arrived: bool) {
        let spell = aura.map(|(spell, _)| spell);
        let rebuilt = std::mem::take(&mut self.camera_rebuilt);
        self.arrival_end = arrived && aura.is_some();
        if arrived || rebuilt {
            self.zoom = None;
        } else if spell != self.aura {
            self.zoom = aura.map(|(_, degrees)| degrees / REFERENCE_DEFAULT_DEGREES);
        }
        self.aura = spell;
    }
}

/// Drives the projection from aura 76 in our own aura slots only, as the reference gates it: the
/// aura field is public, so another player's spyglass must not zoom ours.
pub(super) fn apply_scoped_view(
    self_q: Query<&ObjectStore, With<SelfPlayer>>,
    arrived: Query<(), Added<SelfPlayer>>,
    spells: Option<Res<Spells>>,
    mut scoped: ResMut<ScopedView>,
    mut fov_zoom: ResMut<FovZoom>,
) {
    let catalog = spells.as_ref().map(|s| &s.catalog);
    let aura = self_q
        .single()
        .ok()
        .zip(catalog)
        .and_then(|(store, catalog)| {
            store.0.unit_auras().find_map(|slot| {
                let rec = catalog.get(slot.spell_id)?;
                (0..3).find_map(|i| {
                    // The reference reads the misc value from the aura name's own effect index.
                    (rec.effect_apply_aura[i] == SPELL_AURA_FAR_SIGHT)
                        .then(|| (slot.spell_id, rec.effect_misc_value[i] as f32))
                })
            })
        })
        // A zero misc value is the reference's own "restore" argument, not a zero-width view.
        .filter(|&(_, deg)| deg > 0.0);
    scoped.observe(aura, !arrived.is_empty());

    // `stamp_world_fov` turns it into the field of view at the frame's aspect.
    let want = scoped.zoom.unwrap_or(1.0);
    if fov_zoom.0 != want {
        fov_zoom.0 = want;
    }
}

#[cfg(test)]
mod tests;
