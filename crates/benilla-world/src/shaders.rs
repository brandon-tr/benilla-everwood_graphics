//! The engine's WGSL, embedded in the binary; the game's shaders are embedded by `benilla-app`.
//! These calls must stay in a file directly under `src/`: `embedded_asset!` derives the served
//! path from `file!()` without normalizing it, so a call from a submodule would serve the shader
//! under that module's directory instead of `embedded://benilla_world/shaders/<name>.wgsl`.

use bevy::prelude::*;

/// Embed the engine's WGSL files under `embedded://benilla_world/shaders/`. The first member of
/// [`crate::world_plugins::WorldPlugins`]; it needs `DefaultPlugins`, which creates the asset
/// registry, added first.
pub(crate) fn plugin(app: &mut App) {
    bevy::asset::embedded_asset!(app, "shaders/sky_vertex.wgsl");
    bevy::asset::embedded_asset!(app, "shaders/sky.wgsl");
    // MONKEY (sky): the Enhanced/High library `sky.wgsl` and `cloud.wgsl` import.
    bevy::asset::embedded_asset!(app, "shaders/sky_fx.wgsl");
    bevy::asset::embedded_asset!(app, "shaders/star.wgsl");
    bevy::asset::embedded_asset!(app, "shaders/cloud.wgsl");
    bevy::asset::embedded_asset!(app, "shaders/celestial.wgsl");
    bevy::asset::embedded_asset!(app, "shaders/ffx_glow.wgsl");
    bevy::asset::embedded_asset!(app, "shaders/wow_effect.wgsl");
    bevy::asset::embedded_asset!(app, "shaders/static_gx.wgsl");
    // MONKEY (torch shadows Phase 1): the vertex-only depth-map shader for the interior-fixture pass.
    bevy::asset::embedded_asset!(app, "shaders/torch_depth.wgsl");
    // Temporal depth: the single-sample opaque-depth copy for the future native TAA lane.
    bevy::asset::embedded_asset!(app, "shaders/temporal_depth.wgsl");
    // Temporal motion: camera-only reprojection candidates on that depth.
    bevy::asset::embedded_asset!(app, "shaders/temporal_motion.wgsl");
    // Temporal reactive: opaque-color snapshot and transparent-delta mask.
    bevy::asset::embedded_asset!(app, "shaders/temporal_reactive.wgsl");
    // Temporal rejection: conservative per-view model rejection mask.
    bevy::asset::embedded_asset!(app, "shaders/temporal_rejection.wgsl");
    // Private reconstructed-color resolve and its first-frame seed pass.
    bevy::asset::embedded_asset!(app, "shaders/temporal_resolve.wgsl");
}

#[cfg(test)]
mod tests;
