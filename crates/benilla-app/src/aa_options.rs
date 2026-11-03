//! Native-TAA option and startup camera-sample selection.

use std::collections::HashMap;

use bevy::prelude::*;
use bevy::render::camera::TemporalJitter;
use bevy::render::view::Msaa;

use benilla_world::aa::{
    AaEffective, AaMode, AaPolicy, AaPolicyResolved, AaReadiness, TemporalHistory,
};
use benilla_world::view::WorldCamera;

use crate::cvars::{CvarChanged, Cvars};

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Resource)]
pub(crate) enum AaSelection {
    Off,
    #[default]
    Fallback,
    Native,
}

/// Bevy's own TAA Halton (2, 3) minus 0.5 sequence, in pixels: `[-0.5, 0.5]`, applied in clip/NDC
/// x and y by [`TemporalJitter::jitter_projection`]. Reused rather than invented, so the native
/// lane jitters the same way stock Bevy TAA does.
const HALTON_JITTER: [Vec2; 8] = [
    Vec2::new(0.0, 0.0),
    Vec2::new(0.0, -0.16666666),
    Vec2::new(-0.25, 0.16666669),
    Vec2::new(0.25, -0.3888889),
    Vec2::new(-0.375, -0.055555552),
    Vec2::new(0.125, 0.2777778),
    Vec2::new(-0.125, -0.2777778),
    Vec2::new(0.375, 0.055555582),
];

impl AaSelection {
    pub(crate) fn parse(value: &str) -> Self {
        match value.trim() {
            "0" => Self::Off,
            "2" => Self::Native,
            _ => Self::Fallback,
        }
    }
}

/// The selection applied at camera creation. Runtime CVar writes only change the saved value.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Resource)]
pub(crate) struct AaBootSelection(pub(crate) AaSelection);

pub(crate) fn camera_msaa_for_aa(
    selection: AaSelection,
    fallback: Msaa,
    native_supported: bool,
) -> Msaa {
    match selection {
        AaSelection::Off => Msaa::Off,
        AaSelection::Fallback => fallback,
        AaSelection::Native if native_supported => Msaa::Off,
        AaSelection::Native => fallback,
    }
}

#[derive(SystemSet, Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) struct AaBootApplied;

pub(crate) struct AaOptionsPlugin;

impl Plugin for AaOptionsPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<AaBootSelection>()
            .add_systems(
                Startup,
                (apply_boot_aa_selection, apply_boot_aa_mode)
                    .chain()
                    .after(crate::cvars::CvarLoad)
                    .in_set(AaBootApplied),
            )
            // After the policy has resolved this frame's effective mode, before extraction. The
            // schedule lives here, not in `benilla-world`, so engine GPU fixtures keep explicit
            // control of their jitter.
            .add_systems(Last, schedule_temporal_jitter.after(AaPolicyResolved))
            .add_observer(aa_on_cvar);
    }
}

fn apply_boot_aa_selection(cvars: Res<Cvars>, mut boot: ResMut<AaBootSelection>) {
    if let Some(value) = cvars.get("gxAntiAliasing") {
        boot.0 = AaSelection::parse(value);
    }
}

/// Turn the saved selection into this session's real mode. Native TAA needs both the implementation
/// readiness flag and the policy request; the camera spawn then reads [`AaReadiness`] for its sample
/// count, so a readiness-off selection keeps the MSAA fallback exactly as before. `0` and `1` leave
/// the policy at its existing `Msaa` request and change nothing beyond the camera's sample count.
fn apply_boot_aa_mode(
    boot: Res<AaBootSelection>,
    readiness: Option<ResMut<AaReadiness>>,
    policy: Option<ResMut<AaPolicy>>,
) {
    if boot.0 != AaSelection::Native {
        return;
    }
    if let Some(mut readiness) = readiness {
        readiness.set_native_taa(true);
    }
    if let Some(mut policy) = policy {
        policy.request(AaMode::Taa);
    }
}

/// The camera's `Msaa` for the boot selection, given whether the native implementation is actually
/// ready. Kept beside the mode request so both read the same readiness.
pub(crate) fn boot_camera_msaa(
    selection: AaSelection,
    fallback: Msaa,
    readiness: Option<&AaReadiness>,
) -> Msaa {
    camera_msaa_for_aa(
        selection,
        fallback,
        readiness.is_some_and(|readiness| readiness.native_taa()),
    )
}

/// Schedule the bounded subpixel jitter on the world camera only, while TAA is this frame's
/// effective mode. The sequence restarts whenever the view's history generation moves — a cut, a
/// target change, a mode change or any other invalidation — and the stock [`TemporalJitter`]
/// component is removed as soon as TAA stops being effective. Jitter never touches `Projection`,
/// `Transform`, picking, world-space text or the UI camera.
fn schedule_temporal_jitter(
    policy: Option<Res<AaPolicy>>,
    history: Option<Res<TemporalHistory>>,
    mut commands: Commands,
    cameras: Query<(Entity, Option<&TemporalJitter>), With<WorldCamera>>,
    // `(generation, next index)` per view; `Local` keeps it out of the ECS and out of snapshots.
    mut sequences: Local<HashMap<Entity, (u64, u64)>>,
) {
    let temporal = policy.is_some_and(|policy| policy.effective() == AaEffective::Taa);
    if !temporal {
        sequences.clear();
        for (view, jitter) in &cameras {
            if jitter.is_some() {
                commands.entity(view).remove::<TemporalJitter>();
            }
        }
        return;
    }
    for (view, _) in &cameras {
        let generation = history
            .as_ref()
            .and_then(|history| history.view(view))
            .map_or(0, |history| history.generation());
        let sequence = sequences.entry(view).or_insert((generation, 0));
        if sequence.0 != generation {
            *sequence = (generation, 0);
        }
        let offset = HALTON_JITTER[(sequence.1 as usize) % HALTON_JITTER.len()];
        sequence.1 = sequence.1.wrapping_add(1);
        commands.entity(view).insert(TemporalJitter { offset });
    }
}

fn aa_on_cvar(ev: On<CvarChanged>) {
    if ev.is("gxAntiAliasing") {
        info!("gxAntiAliasing saved; a full client restart applies it");
    }
}

#[cfg(test)]
mod tests;
