//! Which world views demand a native-TAA previous-rig snapshot.
//!
//! Demand is a property of the frame's actual views, not of a stored preference: a saved `Taa`
//! request that resolved to MSAA is not demand, and neither is a `TemporalAntiAliasing` on a
//! portrait booth, a mirror or a UI camera. A view must be a real 3D world camera
//! ([`WorldCamera`] plus [`Camera3d`]; portraits are `Camera3d`s without the marker), must carry
//! [bevy's native TAA mount][TemporalAntiAliasing], and must be compatible with the native prepass
//! the snapshot feeds:
//!
//! - MSAA off (`TemporalAntiAliasing` requires it and the copy must be single-sample);
//! - `DepthPrepass` and `MotionVectorPrepass` present (the component's own `#[require]`s, validated
//!   rather than assumed);
//! - a **resolved** render target with both physical dimensions positive;
//! - no `DeferredPrepass`, no `MainPassResolutionOverride`, and no partial viewport or sub-camera
//!   view.
//!
//! The target size comes from the camera's own resolved target information
//! (`Camera::physical_target_size`, i.e. `Camera.computed.target_info`), not from
//! [`TemporalTarget`](crate::aa::TemporalTarget): `TemporalTarget::new` clamps its dimensions to at
//! least one, so a positive `TemporalTarget` is not evidence that a target exists, and
//! `UVec2::ZERO` can never appear in one. The main-world camera system
//! (`bevy_render::camera::camera_system`) resolves `target_info` in `PostUpdate`/`PostStartup` from
//! the window or the `Assets<Image>` the camera renders into, so by this system's `Last` slot the
//! information is final for the frame; no render-side qualification is needed. Resolved target
//! information is camera state, not a GPU attachment.
//!
//! Attachments are deliberately *not* part of the test. `ViewPrepassTextures`,
//! `TemporalAntiAliasHistoryTextures` and `TemporalAntiAliasing::reset` only exist once a view has
//! already been prepared, and the first demand must be able to seed this service before any of
//! them exists (Bevy's `reset` defaults to `true`).
//!
//! The snapshot itself is world-scope: the palette, the rig tables and the frame block are one
//! shared blob, so several eligible views in one frame still share one copy per epoch. Only the
//! main-world camera ids are retained, and only for diagnostics; nothing here keys per-view state
//! on a main-world entity, so there is no main/render entity confusion to make.

use bevy::anti_alias::taa::TemporalAntiAliasing;
use bevy::camera::{Camera, Camera3d, MainPassResolutionOverride};
use bevy::core_pipeline::prepass::{DeferredPrepass, DepthPrepass, MotionVectorPrepass};
use bevy::prelude::*;
use bevy::render::extract_resource::ExtractResource;
use bevy::render::view::Msaa;

use crate::static_gx::StaticGxMirror;
use crate::view::WorldCamera;

/// The world views that demand this frame's previous-rig snapshot.
///
/// Extracted into the render world so the copy and record stages can tell demand from an idle
/// frame. The ids are main-world cameras kept for diagnostics; the snapshot they share is
/// world-scope.
#[derive(Resource, Clone, Debug, Default, PartialEq, Eq, ExtractResource)]
pub(crate) struct RigTaaWorldDemand {
    main_views: Vec<Entity>,
}

impl RigTaaWorldDemand {
    /// Whether any compatible world view is rendering this frame.
    pub(crate) fn is_empty(&self) -> bool {
        self.main_views.is_empty()
    }
}

/// The compatibility facts one world view reports to the demand scan.
///
/// Split out so the rule is a value, not a query: the tests exercise the same predicate the
/// production system calls.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct RigTaaViewFacts {
    /// The view carries Bevy's native TAA mount.
    pub(crate) native_taa: bool,
    /// `Camera::is_active`.
    pub(crate) camera_active: bool,
    /// The view is a 3D camera (`Camera3d`), not a 2D overlay.
    pub(crate) three_d: bool,
    /// The view renders with `Msaa::Off`.
    pub(crate) msaa_off: bool,
    /// `DepthPrepass` is present.
    pub(crate) depth_prepass: bool,
    /// `MotionVectorPrepass` is present.
    pub(crate) motion_prepass: bool,
    /// `DeferredPrepass` is present; unsupported.
    pub(crate) deferred_prepass: bool,
    /// `MainPassResolutionOverride` is present; unsupported.
    pub(crate) resolution_override: bool,
    /// A partial `Viewport` is set; unsupported.
    pub(crate) viewport: bool,
    /// A `SubCameraView` is set; unsupported.
    pub(crate) sub_view: bool,
    /// The camera's resolved physical render-target extent, when the camera system has one.
    pub(crate) resolved_target: Option<UVec2>,
}

/// Whether a world view is compatible with the native-TAA previous-rig snapshot.
///
/// A target is usable only when it was actually resolved and both physical dimensions are
/// positive; a `(0, h)`, `(w, 0)` or `(0, 0)` target is not.
pub(crate) fn rig_taa_view_is_eligible(facts: RigTaaViewFacts) -> bool {
    facts.native_taa
        && facts.camera_active
        && facts.three_d
        && facts.msaa_off
        && facts.depth_prepass
        && facts.motion_prepass
        && !facts.deferred_prepass
        && !facts.resolution_override
        && !facts.viewport
        && !facts.sub_view
        && facts
            .resolved_target
            .is_some_and(|size| size.x > 0 && size.y > 0)
}

/// Main world (`Last`): publish this frame's eligible world views.
///
/// Runs after the frame's camera, projection and target writers (the camera system's `PostUpdate`
/// resolution included) and before extraction. Ordering between views is normalised so an
/// unchanged set is not republished.
#[allow(clippy::type_complexity)] // Bevy query tuples read better inline.
pub(crate) fn extract_rig_taa_world_views(
    views: Query<
        (
            Entity,
            &Camera,
            &Msaa,
            Has<Camera3d>,
            Has<TemporalAntiAliasing>,
            Has<DepthPrepass>,
            Has<MotionVectorPrepass>,
            Has<DeferredPrepass>,
            Has<MainPassResolutionOverride>,
        ),
        (With<WorldCamera>, Without<StaticGxMirror>),
    >,
    mut demand: ResMut<RigTaaWorldDemand>,
) {
    let mut main_views = Vec::new();
    for (view, camera, msaa, three_d, native_taa, depth, motion, deferred, resolution_override) in
        &views
    {
        let facts = RigTaaViewFacts {
            native_taa,
            camera_active: camera.is_active,
            three_d,
            msaa_off: *msaa == Msaa::Off,
            depth_prepass: depth,
            motion_prepass: motion,
            deferred_prepass: deferred,
            resolution_override,
            viewport: camera.viewport.is_some(),
            sub_view: camera.sub_camera_view.is_some(),
            resolved_target: camera.physical_target_size(),
        };
        if rig_taa_view_is_eligible(facts) {
            main_views.push(view);
        }
    }
    main_views.sort_unstable_by_key(|view| view.to_bits());
    if demand.main_views != main_views {
        demand.main_views = main_views;
    }
}

#[cfg(test)]
mod tests;
