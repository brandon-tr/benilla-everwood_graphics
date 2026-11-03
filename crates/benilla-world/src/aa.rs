//! Anti-aliasing policy and the temporal-history lifecycle native TAA uses.
//!
//! The default request is the established MSAA path and `gxAntiAliasing = 1` keeps it unchanged;
//! `gxAntiAliasing = 2` asks for native TAA instead (see [`AaReadiness`] and the app's boot
//! selection). Only an effective temporal view runs any temporal work. The private `depth` child
//! publishes a per-view current/previous opaque-depth pair, the private `motion` child turns that
//! into camera-only reprojection candidates, the private `reactive` child publishes a post-SSAO
//! opaque-color snapshot plus a transparent/transmissive changed-color mask, the private
//! `skin_bounds` child publishes conservative current-pose bounds for skinned model meshes, and
//! the private `rejection` child publishes per-view projected model rejection masks, including
//! selected retained static-gx wind foliage through coarse expanded regions, and the private
//! `scene_color` child retains current/previous assembled scene-color captures before downstream
//! atmospheric compositing, and the private `resolve` child reconstructs only cameras carrying the
//! [`TemporalResolveView`] marker into independently owned history. [`AaPlugin`] inserts that
//! marker on the world camera while the effective mode is TAA and removes it otherwise, schedules
//! the bounded subpixel jitter, and bridges a successfully encoded resolve back into
//! [`TemporalHistory`]. Camera-only candidates are not object/deformation motion, mask zero is not
//! sufficient confidence, the skin bounds are metadata for screen-space rejection, and rejection
//! zero does not authorize reuse. Depth and mask pairs retain current and immediately previous
//! data, but that is not production color/depth reconstruction history. Object motion, deformation
//! motion and temporal upscaling remain deferred.
//!
//! # The policy
//!
//! [`AaMode`] is a request, not a promise:
//!
//! - `Off`: native resolution, no AA. It is effective only when the camera's actual sample count is
//!   1. Above 1 this foundation does not apply an MSAA change, so the request falls back to the
//!   existing `Msaa` path with [`AaReason::OffUnavailable`].
//! - `Msaa`: the existing GPU-clamped `gxMultisample` path ([`crate::view::MsaaSetting`], the
//!   camera's [`Msaa`] component). This is the default request and the fallback for every
//!   unavailable mode.
//! - `Taa`: native-resolution temporal reconstruction through the fork's own resolve lane. It is
//!   effective only while [`AaReadiness`] says the native implementation is selected; otherwise the
//!   request falls back to the MSAA path with [`AaReason::TaaUnavailable`]. The app pairs an
//!   effective TAA with `Msaa::Off`, and the temporal lanes decline a multisampled view.
//! - `TemporalUpscale`: unavailable until separately implemented and verified.
//!
//! [`AaPolicy`] resolves a request to an [`AaEffective`] mode and an [`AaReason`]. The effective
//! mode is what the renderer does; it is never `Taa` or `TemporalUpscale` while the corresponding
//! [`AaReadiness`] flag is false. A temporal request resolves to `Msaa` with a `...Unavailable`
//! reason, never silently to active TAA. `gxAntiAliasing` stays the only persistent AA setting: the
//! app reads it once at startup and requests the matching mode, and a live write still needs a
//! client restart because the camera's sample count is fixed at spawn.
//!
//! Any graphical feature must ship with an options-menu Off control that stops its work and
//! invalidates its history. Native TAA's control is the existing `gxAntiAliasing` row: `0` and `1`
//! are the established paths and `2` selects TAA at the next client start.
//!
//! The policy is resolved at [`Last`], after the frame's `MsaaSetting`/camera writes and before
//! extraction, so an inspectable [`AaPolicy`] always names the frame's real mode.
//!
//! Deviation: native TAA and temporal upscaling are modern graphics the 1.12.1 client has no
//! mechanism for; they are an Everwood fork feature behind `gxAntiAliasing = 2`, absent from the
//! stock UI and never the 1.12.1 default.
//!
//! # The history
//!
//! [`TemporalHistory`] is a per-world-view record, not a settings store. A view is registered only
//! from the [`WorldCamera`] lifecycle; portrait booths, reflection probes, the minimap and the
//! player-UI camera are separate `Camera3d`/`Camera2d` entities and never enter it. Each slot is
//! tied to a [`TemporalTarget`] (physical size and render scale) and a generation:
//!
//! - `begin` issues a [`HistoryToken`] for one frame. It does not make history valid.
//! - `commit` is the only transition to valid, and only for the token of the current generation
//!   that is still in flight. Call it after the reconstruction and the history colour/depth writes
//!   have succeeded.
//! - `abandon` or a begin followed by no commit drops the slot: a skipped frame, a failed pipeline,
//!   an allocation failure or a frame that never reached the resolve must not leave old history
//!   reusable as if it were one frame old.
//! - `invalidate` bumps the generation, clears validity and drops any in-flight token, so work
//!   from an older generation can never commit. Lifecycle hooks below call it; a stale commit is
//!   rejected rather than resurrecting history.
//!
//! The caller owns the GPU textures. This state says whether a previous pair exists and whether it
//! may be sampled; it deliberately does not allocate, so a simulation tick or a pipeline compile
//! cannot mark history valid. Previous/current ownership: the current frame resolves into the
//! view's main texture and writes a separate history write texture, sampling only the history read
//! texture (a ping-pong pair); a failed write leaves the read pair untouched, and only a successful
//! write commits it. The current frame never samples the texture it is writing.
//!
//! # Pinned conventions for the later motion/depth and TAA work
//!
//! These are read from the pinned Bevy 0.18.1 sources, not assumed:
//!
//! - **Space and depth.** The camera is right-handed and looks down `-Z`; `benilla-world` uses
//!   `PerspectiveProjection` with `Mat4::perspective_infinite_reverse_rh` (`bevy_camera 0.18.1`,
//!   `projection.rs`), so NDC depth is reversed: near = 1, far = 0, and the far plane is infinite
//!   for raster depth while [`crate::view::CAM_FAR`] remains the culling far plane.
//! - **Jitter.** `TemporalJitter.offset` is a subpixel offset in pixels in `[-0.5, 0.5]` and is
//!   applied in clip/NDC x and y (`bevy_render 0.18.1`, `camera.rs`). Jitter must live only in the
//!   world view's extracted clip matrix and the history's previous/current projection; it must not
//!   mutate `Projection`, `Transform` or the camera's transform. Picking, UI layout, world-space
//!   text and every screen-space overlay use the unjittered projection. The motion child records
//!   each frame's actual jittered projection plus the UV jitter offset; the [`native`] scheduler owns
//!   the sequence and resets it whenever the view's history generation moves.
//! - **Motion vectors.** Bevy's prepass writes `(current_ndc - previous_ndc) * (0.5, -0.5)` in UV
//!   units, i.e. current UV minus previous UV with V downwards (`bevy_pbr 0.18.1`,
//!   `prepass.wgsl` / `pbr_prepass_functions.wgsl`). The current side uses
//!   `unjittered_clip_from_world`, never the jittered matrix. History reprojection is
//!   `history_uv = current_uv - motion_vector`, exactly Bevy's own TAA shader. Everwood's motion
//!   child instead reconstructs with the actual jittered current/previous projections for its raw
//!   depth/mask checks, then subtracts the jitter delta so the published vector is unjittered.
//! - **Matrix ownership.** Keep current/previous `view_from_world`, the actual jittered
//!   `clip_from_view` used by the copied raster, and each frame's UV jitter offset. Those actual
//!   matrices validate the copied depth and reactive mask at raw previous raster coordinates. The
//!   model rejection lane remains conservative padded CPU-projected bounds, not rasterized
//!   jittered geometry. The output motion is corrected to the unjittered convention; the UI and
//!   picking get the unjittered projection. The temporal-depth copy in `depth.rs` preserves the
//!   scene's actual depth raster; it is not an independently unjittered depth image.
//!
//! # The graph and the proposed TAA boundary
//!
//! The world camera renders `Rgba16Float` with `Hdr`, `Tonemapping::None` and
//! `CameraOutputMode::Skip`; its main texture is gamma-valued scene colour, not linear HDR
//! (`benilla_app::world_backdrop` documents the byte-space combine). The current `Core3d` order is:
//!
//! `StartMainPass -> StaticGxLabel -> MainOpaquePass -> AoLabel -> WaterDepthLabel ->
//! MainTransparentPass -> TemporalSceneColorLabel -> EndMainPass -> FogLabel -> VolLightLabel ->
//! StartMainPassPostProcessing -> ShaftLabel -> BloomLabel -> GradeLabel -> FfxGlowLabel ->
//! Node3d::Bloom/tonemapping`.
//!
//! There is no Bevy normal/depth prepass today: SSAO reconstructs from the main-pass depth and the
//! retained static world draws in its own node, so Bevy's TAA/prepass plugins cannot be assumed to
//! cover either path. The TAA work needs its own depth/motion prepass or an equivalent publication
//! from these nodes.
//!
//! The native reconstruction boundary is a `Core3d` node after motion and before `EndMainPass`:
//! `(TemporalMotionLabel, TemporalSceneColorLabel, EndMainPass)`. Its current/previous history is
//! the assembled gamma-valued scene color before the app-owned volumetric fog/light composites,
//! which continue rendering from the current frame downstream. This keeps atmospheric output with
//! no established temporal correspondence out of scene history. It requires MSAA off (a temporal
//! resolve cannot read a multisampled color attachment directly), a resolved depth attachment
//! (`Depth32Float`, reverse), and motion vectors; this task enables none of them. The color capture
//! itself is not reconstructed output or valid committed history, and the player UI remains outside
//! world history. Temporal upscaling would be a different, later node with a smaller internal render
//! target and is not implied by this boundary.
//!
//! Transparency is the hard edge: the depth buffer is the opaque depth, with the water lane copying
//! it before transparents. Transparent colour is not opaque depth. Water, particles, ribbons and
//! other blended world geometry have no reliable velocity in the current graph; until they write
//! velocity or render after the resolve, the resolve must reject them by depth/colour (or draw them
//! after it), never treat their colour as a depth sample.
//!
//! # Coverage the later work must provide
//!
//! | Lane | Current path | Motion needed |
//! | --- | --- | --- |
//! | Static retained world/doodads | `StaticGxLabel`, retained buffers | camera-only for rigid statics; wind-deformed foliage is not rigid |
//! | Terrain/WMO | `MainOpaquePass`, `terrain.wgsl`/`static_gx.wgsl` | camera-only for rigid surfaces |
//! | Moving/skinned actors, weapons | `wow_model.wgsl`, `rig_palette` | per-vertex previous skin/pose; Bevy `SkinnedMesh`/prepass is not used |
//! | Wind grass/trees | `wind_hook.wgsl` in `wow_model.wgsl` and `static_gx.wgsl` | vertex/object velocity or rejection; camera-only is wrong |
//! | Water | `liquid` between opaque and transparent, depth copied | displaced, transparent, no depth write: reject or draw after resolve |
//! | Particles/ribbons | transparent after `MainTransparentPass` | no reliable velocity: reject or draw after resolve |
//! | Sky/distant horizon | `sky`/`wdl`, depth pinned | camera-only background; far-depth rejection |
//! | Nameplates, chat bubbles, combat text, raid marks | world camera, camera-facing blended glyph meshes | world-space overlays with no reliable velocity; exclude or draw after the resolve, never an opaque-depth donor |
//! | HUD, text, icons, cursor | player-UI camera, `Camera2d` | never in world history; UI stays unjittered |
//!
//! Screen overlays are not world-space geometry. The player-UI camera, its FFX combine and all
//! `Camera2d` content (HUD, icons, text, cursor) must never sample or write world history.
//! Nameplates, chat bubbles, combat text and raid marks are the opposite case: they are camera-
//! facing **world-space** glyph meshes drawn in the world pass, depth-tested but alpha-blended with
//! no depth write, so they are considered with the transparent lane and excluded/rejected or drawn
//! after the resolve; they are not UI-camera content.
//!
//! # Rejection
//!
//! For content without reliable velocity, reject rather than accumulate: a depth/normal test
//! against the current scene, a bounded motion-vector test, and colour-space (YCoCg) variance
//! clipping around the current 3x3 neighbourhood are the planned tools. An off-screen or
//! zero-confidence reprojection resets to the current sample. A transparent fragment is never a
//! valid opaque-depth donor.
//!
//! # Lifecycle wiring
//!
//! `AaPlugin` runs at [`Last`], after the frame's writers and before extraction. Implemented here:
//! view add/remove (`WorldCamera`), target size/scale changes (`TemporalTarget`), projection
//! changes, camera activity changes, `WorldLive` flips, the loading cover (`Viewer::world_covered`),
//! cross-map `MapChange`, and effective AA-mode changes. A same-map `TeleportMessage` is wired by
//! `benilla-app`'s player plugin. Explicit [`AaInvalidation::CameraCut`] and
//! [`TemporalHistory::invalidate_all`] are the API for a cinematic cut or any producer that does not
//! have a component/message yet.
//!
//! Invalidation is value-based, not change-tick-based. `Changed<Projection>`/`Changed<Camera>` only
//! nominate a candidate: history compares the unjittered clip matrix and the explicit camera values
//! (activity, viewport position/size/depth, sub-camera view). A same-value write, including Bevy's
//! computed camera bookkeeping, preserves committed history; only an actual difference invalidates.
//! Snapshots are seeded on first observation and stored even while history is invalid, so steady
//! frames settle.
//!
//! Implemented here as well: the production marker insertion/removal on the world camera
//! ([`TemporalResolveView`], keyed on the effective mode), the automatic bounded Halton jitter on
//! that camera only, and the renderer-to-CPU history bridge. [`native`] settles a begun token only
//! after a later extraction proves the resolve's render frame completed; an encoded resolve commits,
//! a skipped, failed, pipeline-less or invalidated frame abandons. The bridge never uses wall-clock
//! ordering: extraction epochs and the resolve's encode stamp are the only ordering evidence.
//!
//! Deferred, and not claimed: object and deformation motion (the current candidate is
//! camera-only), temporal upscaling, a cinematic-cut producer beyond [`AaInvalidation::CameraCut`]
//! (the API exists, nothing generates a cut message), and host-GPU pixel validation (the 13
//! `#[ignore]` readback tests remain the host gate). The UI/picking projections stay unjittered
//! because jitter lives only in the world camera's extracted clip matrix, and the player-UI camera
//! and every `Camera2d` are separate views that never carry the marker or the jitter. The private
//! resolve blends only marker cameras and owns separate reconstructed history; raw scene-color
//! captures remain a separate diagnostic lane. The reactive mask
//! detects visible color changes only; it does not classify geometry or every transparent fragment,
//! and it retains only a current/previous mask pair. The resolve samples the previous mask at
//! reprojected UV, and later atmospheric passes remain outside its coverage. The depth lane
//! retains current and immediately previous reverse-Z captures; previous depth still must be
//! compared against expected previous depth before history reuse. The skin bounds feed this frame's
//! projected model rejection rectangles. Retained static-gx wind foliage is covered through
//! coarse expanded region bounds, not velocity or per-leaf tightness; nearby rigid pixels may also
//! be rejected. Animated materials and later atmospheric passes remain uncovered. Neither lane
//! implements object motion or final GPU history acceptance, and rejection zero is not proof of
//! temporal safety. Depth and mask pairs do not create color/depth reconstruction history. The
//! resolve uses current AND reprojected previous rejection masks.

use std::collections::HashMap;

use bevy::camera::SubCameraView;
use bevy::prelude::*;
use bevy::render::view::Msaa;

use crate::schedule::WorldLive;
use crate::view::{MsaaSetting, Viewer, WorldCamera};
use crate::world_map::MapChange;

mod depth;
mod frame_pair;
mod motion;
mod native;
mod reactive;
mod rejection;
mod resolve;
mod scene_color;
mod skin_bounds;

#[cfg(test)]
mod bevy_reference;
#[cfg(test)]
mod gpu_validation;

pub use reactive::TemporalOpaqueSnapshotLabel;
pub use resolve::TemporalResolveView;

/// A requested anti-aliasing mode. `Msaa` is the existing fallback; `Taa` is the fork's native
/// temporal lane, gated on [`AaReadiness`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AaMode {
    /// Native resolution with no anti-aliasing. Effective only when the camera already renders with
    /// one sample; this policy does not change MSAA to implement it.
    Off,
    /// The existing, GPU-clamped `gxMultisample` path.
    Msaa,
    /// Native-resolution temporal reconstruction through the fork's own resolve lane.
    Taa,
    /// Temporal upscaling, unavailable until separately implemented and verified.
    TemporalUpscale,
}

impl AaMode {
    /// Whether this mode accumulates temporal history.
    pub fn is_temporal(self) -> bool {
        matches!(self, AaMode::Taa | AaMode::TemporalUpscale)
    }
}

/// What the renderer does this frame. `Taa` is effective only while [`AaReadiness`] allows it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AaEffective {
    /// No AA, at native resolution.
    Off,
    /// The existing MSAA path with this many samples (1 means the `Msaa` component is off).
    Msaa { samples: u32 },
    /// Native-resolution temporal reconstruction.
    Taa,
    /// Temporal upscaling.
    TemporalUpscale,
}

impl AaEffective {
    /// Whether this mode accumulates temporal history.
    pub fn is_temporal(self) -> bool {
        matches!(self, AaEffective::Taa | AaEffective::TemporalUpscale)
    }
}

/// Why a request resolved to an effective mode. Inspectable, never a hidden fallback.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AaReason {
    OffRequested,
    /// Off was requested but the camera still renders with multisampling; this foundation does not
    /// apply an MSAA change, so the existing path remains effective.
    OffUnavailable,
    MsaaRequested,
    TaaRequested,
    TemporalUpscaleRequested,
    /// TAA was requested but native readiness is off; the existing MSAA path is used.
    TaaUnavailable,
    /// Temporal upscaling was requested but is not implemented; the existing MSAA path is used.
    TemporalUpscaleUnavailable,
}

/// What the implementation is ready to do. Defaults are false so an unfinished temporal mode is
/// never advertised by accident. `benilla-app` sets the TAA flag when the boot selection asks for
/// native TAA; this says the implementation is wired, never that a particular host GPU has been
/// pixel-validated.
#[derive(Resource, Default, Clone, Copy, Debug, PartialEq, Eq)]
pub struct AaReadiness {
    pub(crate) taa: bool,
    pub(crate) temporal_upscale: bool,
}

impl AaReadiness {
    /// Record whether the native TAA implementation is selected. The app calls this from its boot
    /// selection; a false flag makes a TAA request resolve to the MSAA fallback.
    pub fn set_native_taa(&mut self, ready: bool) {
        self.taa = ready;
    }

    /// Whether a TAA request may become effective.
    pub fn native_taa(self) -> bool {
        self.taa
    }

    /// The same value as an explicit readiness record, for a caller that has no resource yet.
    pub fn with_native_taa(ready: bool) -> Self {
        Self {
            taa: ready,
            temporal_upscale: false,
        }
    }
}

/// The frame's requested and effective anti-aliasing mode, with the reason for any fallback.
///
/// This is transient state, not a setting: `gxAntiAliasing` and `MsaaSetting` keep their existing
/// persistence and restart semantics. The app requests the boot mode once; nothing re-resolves a
/// live CVar write.
#[derive(Resource, Clone, Copy, Debug, PartialEq, Eq)]
pub struct AaPolicy {
    requested: AaMode,
    effective: AaEffective,
    reason: AaReason,
}

impl Default for AaPolicy {
    /// The existing path: the default `gxMultisample` is 1, which `Msaa` reads as off.
    fn default() -> Self {
        Self {
            requested: AaMode::Msaa,
            effective: AaEffective::Msaa { samples: 1 },
            reason: AaReason::MsaaRequested,
        }
    }
}

impl AaPolicy {
    /// The existing path with `samples` from the configuration, used when the plugin installs the
    /// resource so an inspector before the first camera already reads the real `gxMultisample`.
    fn seeded(samples: u32) -> Self {
        Self {
            requested: AaMode::Msaa,
            effective: AaEffective::Msaa { samples },
            reason: AaReason::MsaaRequested,
        }
    }
    /// The mode a caller asked for.
    pub fn requested(self) -> AaMode {
        self.requested
    }

    /// What the renderer can do this frame.
    pub fn effective(self) -> AaEffective {
        self.effective
    }

    /// Why the request resolved this way.
    pub fn reason(self) -> AaReason {
        self.reason
    }

    /// Ask for `mode`. It is resolved at the next [`Last`] pass; this never changes rendering by
    /// itself. `benilla-app` calls it once from the `gxAntiAliasing` boot selection.
    pub fn request(&mut self, mode: AaMode) {
        self.requested = mode;
    }

    /// Resolve `requested` against the GPU-clamped sample count and the implementation readiness.
    /// An unavailable temporal mode always falls back to `Msaa`, never to a temporal effective.
    /// `Off` is effective only at one actual sample; above that it reports `Msaa`/`OffUnavailable`
    /// because this foundation does not apply an MSAA change.
    pub fn resolve(
        requested: AaMode,
        msaa_samples: u32,
        readiness: AaReadiness,
    ) -> (AaEffective, AaReason) {
        match requested {
            AaMode::Off if msaa_samples <= 1 => (AaEffective::Off, AaReason::OffRequested),
            AaMode::Off => (
                AaEffective::Msaa {
                    samples: msaa_samples,
                },
                AaReason::OffUnavailable,
            ),
            AaMode::Msaa => (
                AaEffective::Msaa {
                    samples: msaa_samples,
                },
                AaReason::MsaaRequested,
            ),
            AaMode::Taa if readiness.taa => (AaEffective::Taa, AaReason::TaaRequested),
            AaMode::Taa => (
                AaEffective::Msaa {
                    samples: msaa_samples,
                },
                AaReason::TaaUnavailable,
            ),
            AaMode::TemporalUpscale if readiness.temporal_upscale => (
                AaEffective::TemporalUpscale,
                AaReason::TemporalUpscaleRequested,
            ),
            AaMode::TemporalUpscale => (
                AaEffective::Msaa {
                    samples: msaa_samples,
                },
                AaReason::TemporalUpscaleUnavailable,
            ),
        }
    }
}

/// The target configuration a view's history belongs to: physical pixels and the render scale.
///
/// A rebuilt target, a window resize, a render-scale change or a DPI move changes this component;
/// [`AaPlugin`] invalidates that view's history on the change. The identity is deliberately the
/// resolved size and scale, not a transient GPU handle: the renderer may recreate its textures
/// without changing the logical target.
#[derive(Component, Clone, Copy, Debug, PartialEq, Eq)]
pub struct TemporalTarget {
    size: UVec2,
    scale_bits: u32,
}

impl Default for TemporalTarget {
    fn default() -> Self {
        Self {
            size: UVec2::ONE,
            scale_bits: 1.0f32.to_bits(),
        }
    }
}

impl TemporalTarget {
    /// A target of `size` physical pixels rendered at `scale`.
    pub fn new(size: UVec2, scale: f32) -> Self {
        Self {
            size: UVec2::new(size.x.max(1), size.y.max(1)),
            scale_bits: scale.to_bits(),
        }
    }

    /// The target's physical size in pixels.
    pub fn size(self) -> UVec2 {
        self.size
    }

    /// The target's render scale.
    pub fn scale(self) -> f32 {
        f32::from_bits(self.scale_bits)
    }
}

/// Why a view's temporal history was invalidated, kept for inspection and tests.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AaInvalidation {
    /// The view first appeared; no history has ever been written.
    InitialUse,
    /// The view entity was replaced (a new `WorldCamera` or a removed one).
    CameraReplaced,
    /// An explicit camera cut (cinematic or a producer without a component/message yet).
    CameraCut,
    /// A same-map teleport (`benilla_app`'s player plugin wires its message).
    Teleport,
    /// A fresh world (login, cross-map, or `WorldLive` rising).
    WorldTransition,
    /// The world was left; its streamed state is torn down.
    WorldLeft,
    /// The loading cover rose or fell.
    LoadingCover,
    /// The target size or render scale changed.
    TargetChanged,
    /// The projection changed (near/far/fov/aspect/orthographic mode).
    ProjectionChanged,
    /// The effective AA mode changed.
    AaModeChanged,
    /// Rendering was interrupted by a gated/inactive view or a skipped pipeline.
    RenderingInterrupted,
    /// A frame began but was neither committed nor abandoned before the next begin.
    FrameNotCommitted,
    /// A reconstruction/write failed and the frame was abandoned.
    FrameAbandoned,
    /// The camera became inactive (the loading screen's gate or a glue screen).
    ViewInactive,
    /// The camera's viewport or activity changed.
    ViewportChanged,
}

/// One frame's permission to commit history. It names the generation it was issued for, so work
/// that finishes after an invalidation cannot mark newer history valid.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct HistoryToken {
    view: Entity,
    generation: u64,
    frame: u64,
    had_history: bool,
}

impl HistoryToken {
    pub fn view(self) -> Entity {
        self.view
    }

    pub fn generation(self) -> u64 {
        self.generation
    }

    pub fn frame(self) -> u64 {
        self.frame
    }

    /// Whether the view had a valid previous history when this frame began.
    pub fn had_history(self) -> bool {
        self.had_history
    }
}

/// The unjittered clip matrix a view last reported. A writer is free to take `&mut Projection`
/// every frame (as `apply_scoped_view` does); only a different matrix is a real change.
#[derive(Clone, Copy, Debug)]
struct ProjectionSnapshot {
    clip: Mat4,
}

impl ProjectionSnapshot {
    fn new(projection: &Projection) -> Self {
        Self {
            clip: projection.get_clip_from_view(),
        }
    }

    fn is_finite(self) -> bool {
        self.clip.to_cols_array().iter().all(|v| v.is_finite())
    }
}

impl PartialEq for ProjectionSnapshot {
    fn eq(&self, other: &Self) -> bool {
        self.clip == other.clip
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct ViewportSnapshot {
    position: UVec2,
    size: UVec2,
    depth_start_bits: u32,
    depth_end_bits: u32,
}

/// The explicit `Camera` values a temporal history depends on. `Camera::computed` and the rest of
/// the component's bookkeeping are deliberately ignored: Bevy's `camera_system` rewrites them when
/// the projection changes, and they must not invalidate history on their own.
#[derive(Clone, Copy, Debug, PartialEq)]
struct CameraSnapshot {
    is_active: bool,
    viewport: Option<ViewportSnapshot>,
    sub_camera_view: Option<SubCameraView>,
}

impl CameraSnapshot {
    fn new(camera: &Camera) -> Self {
        Self {
            is_active: camera.is_active,
            viewport: camera.viewport.as_ref().map(|viewport| ViewportSnapshot {
                position: viewport.physical_position,
                size: viewport.physical_size,
                depth_start_bits: viewport.depth.start.to_bits(),
                depth_end_bits: viewport.depth.end.to_bits(),
            }),
            sub_camera_view: camera.sub_camera_view,
        }
    }
}

/// One view's history state, without the GPU textures themselves.
#[derive(Clone, Copy, Debug)]
pub struct ViewHistory {
    target: TemporalTarget,
    generation: u64,
    valid: bool,
    in_flight: Option<HistoryToken>,
    last_commit_frame: Option<u64>,
    last_invalidation: AaInvalidation,
    /// The last projection/camera values this view reported, seeded on first observation and
    /// retained across invalidation so equal rewritten components settle.
    projection: Option<ProjectionSnapshot>,
    camera: Option<CameraSnapshot>,
}

impl ViewHistory {
    fn new(target: TemporalTarget) -> Self {
        Self {
            target,
            generation: 1,
            valid: false,
            in_flight: None,
            last_commit_frame: None,
            last_invalidation: AaInvalidation::InitialUse,
            projection: None,
            camera: None,
        }
    }

    pub fn target(self) -> TemporalTarget {
        self.target
    }

    pub fn generation(self) -> u64 {
        self.generation
    }

    /// Whether a completed history pair may be sampled.
    pub fn is_valid(self) -> bool {
        self.valid
    }

    /// The frame whose reconstruction was last committed.
    pub fn last_commit_frame(self) -> Option<u64> {
        self.last_commit_frame
    }

    pub fn last_invalidation(self) -> AaInvalidation {
        self.last_invalidation
    }

    /// Observe the projection's value. `Changed<Projection>` only nominates a candidate; an equal
    /// finite matrix, or the first observation, preserves history. A different or non-finite matrix
    /// invalidates, and the new value is kept even while history stays invalid so later equal
    /// frames settle.
    fn observe_projection(&mut self, projection: &Projection) {
        let snapshot = ProjectionSnapshot::new(projection);
        match self.projection {
            None => {
                self.projection = Some(snapshot);
                if !snapshot.is_finite() {
                    self.invalidate(AaInvalidation::ProjectionChanged);
                }
            }
            Some(previous) if previous != snapshot || !snapshot.is_finite() => {
                self.projection = Some(snapshot);
                self.invalidate(AaInvalidation::ProjectionChanged);
            }
            Some(_) => {}
        }
    }

    /// Observe the explicit camera values. Computed values are ignored. Activity transitions report
    /// `ViewInactive` when disabling and `RenderingInterrupted` when re-enabling; any other real
    /// change is a viewport/sub-view change. The first observation only seeds.
    fn observe_camera(&mut self, camera: &Camera) {
        let snapshot = CameraSnapshot::new(camera);
        let Some(previous) = self.camera else {
            self.camera = Some(snapshot);
            return;
        };
        if previous == snapshot {
            return;
        }
        let reason = if previous.is_active != snapshot.is_active {
            if snapshot.is_active {
                AaInvalidation::RenderingInterrupted
            } else {
                AaInvalidation::ViewInactive
            }
        } else {
            AaInvalidation::ViewportChanged
        };
        self.camera = Some(snapshot);
        self.invalidate(reason);
    }

    fn invalidate(&mut self, reason: AaInvalidation) {
        self.generation = self.generation.wrapping_add(1);
        if self.generation == 0 {
            self.generation = 1;
        }
        self.valid = false;
        self.in_flight = None;
        self.last_invalidation = reason;
    }
}

/// The per-view temporal-history state. See the module docs for the commit protocol.
#[derive(Resource, Default)]
pub struct TemporalHistory {
    views: HashMap<Entity, ViewHistory>,
    next_frame: u64,
}

impl TemporalHistory {
    /// Register a view with its target, or update an existing registration. A changed target
    /// invalidates that view; no other view is touched.
    pub fn register(&mut self, view: Entity, target: TemporalTarget) {
        match self.views.get_mut(&view) {
            Some(history) => {
                if history.target != target {
                    history.invalidate(AaInvalidation::TargetChanged);
                    history.target = target;
                }
            }
            None => {
                self.views.insert(view, ViewHistory::new(target));
            }
        }
    }

    /// Drop a view's history when its camera is removed.
    pub fn unregister(&mut self, view: Entity) -> bool {
        self.views.remove(&view).is_some()
    }

    pub fn is_registered(&self, view: Entity) -> bool {
        self.views.contains_key(&view)
    }

    /// The frame's in-flight permission, when one was begun and not yet committed or abandoned.
    pub fn in_flight(&self, view: Entity) -> Option<HistoryToken> {
        self.views.get(&view).and_then(|history| history.in_flight)
    }

    /// The view's state, if it was registered.
    pub fn view(&self, view: Entity) -> Option<&ViewHistory> {
        self.views.get(&view)
    }

    /// Whether a previous pair may be sampled for this view.
    pub fn is_valid(&self, view: Entity) -> bool {
        self.views.get(&view).is_some_and(|h| h.valid)
    }

    /// Update the target configuration; a real change invalidates the view.
    pub fn set_target(&mut self, view: Entity, target: TemporalTarget) {
        self.register(view, target);
    }

    /// Begin one frame's reconstruction. `None` for a view that was never registered. A previous
    /// begin that never committed or abandoned is treated as a skipped frame and invalidates the
    /// old pair before this frame gets its token.
    pub fn begin(&mut self, view: Entity) -> Option<HistoryToken> {
        let history = self.views.get_mut(&view)?;
        if history.in_flight.is_some() {
            history.invalidate(AaInvalidation::FrameNotCommitted);
        }
        self.next_frame = self.next_frame.wrapping_add(1);
        if self.next_frame == 0 {
            self.next_frame = 1;
        }
        let token = HistoryToken {
            view,
            generation: history.generation,
            frame: self.next_frame,
            had_history: history.valid,
        };
        history.in_flight = Some(token);
        Some(token)
    }

    /// Mark the frame's history valid. Only the current generation's in-flight token succeeds;
    /// stale, duplicate or foreign-view tokens return `false` and change nothing.
    pub fn commit(&mut self, token: HistoryToken) -> bool {
        let Some(history) = self.views.get_mut(&token.view) else {
            return false;
        };
        if history.generation != token.generation || history.in_flight != Some(token) {
            return false;
        }
        history.valid = true;
        history.in_flight = None;
        history.last_commit_frame = Some(token.frame);
        true
    }

    /// Drop this frame's work (a skipped or failed reconstruction). Only the current generation's
    /// in-flight token can be abandoned; the next begin gets no previous history to sample.
    pub fn abandon(&mut self, token: HistoryToken) -> bool {
        let Some(history) = self.views.get_mut(&token.view) else {
            return false;
        };
        if history.generation != token.generation || history.in_flight != Some(token) {
            return false;
        }
        history.invalidate(AaInvalidation::FrameAbandoned);
        true
    }

    /// Process an explicit invalidation notification for one view: drop any in-flight work, bump
    /// its generation and forbid reuse. Every call advances a registered view; callers suppress
    /// unchanged state before notifying. An unregistered view is a no-op.
    pub fn invalidate(&mut self, view: Entity, reason: AaInvalidation) {
        let Some(history) = self.views.get_mut(&view) else {
            return;
        };
        history.invalidate(reason);
    }

    /// Process a world-scope invalidation notification for every registered view. Every call
    /// advances every registered view; callers suppress unchanged state before notifying.
    pub fn invalidate_all(&mut self, reason: AaInvalidation) {
        for history in self.views.values_mut() {
            history.invalidate(reason);
        }
    }
}

/// The set holding this frame's policy resolution. The app's jitter schedule (and any other
/// consumer that must act on the frame's effective mode before extraction) orders after it.
#[derive(SystemSet, Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct AaPolicyResolved;

/// The AA policy/history plugin: resolve the policy, wire the lifecycle signals available in
/// `benilla-world`, and run the native-TAA production wiring (marker, jitter, history bridge). The
/// app adds the same-map teleport hook and the boot selection.
pub struct AaPlugin;

impl Plugin for AaPlugin {
    fn build(&self, app: &mut App) {
        let samples = app
            .world()
            .get_resource::<MsaaSetting>()
            .map_or(1, |setting| setting.samples);
        app.configure_sets(Last, AaPolicyResolved);
        let bridge = native::NativeResolveBridge::default();
        app.insert_resource(AaPolicy::seeded(samples))
            .init_resource::<AaReadiness>()
            .init_resource::<TemporalHistory>()
            .insert_resource(bridge.clone())
            .add_systems(
                Last,
                (
                    track_world_views,
                    track_temporal_target,
                    track_projection,
                    track_camera,
                    track_world_live,
                    track_loading_cover,
                    track_map_changes,
                    resolve_policy.in_set(AaPolicyResolved),
                    // The effective mode is final here: mark the camera, then begin/settle CPU
                    // history, all before extraction.
                    native::sync_native_resolve_marker,
                    native::maintain_native_history,
                )
                    .chain(),
            );
        // The resolve's render half publishes into this same channel; both worlds share one `Arc`.
        if let Some(render_app) = app.get_sub_app_mut(bevy::render::RenderApp) {
            render_app.insert_resource(bridge);
        }
        // The temporal-depth capture is a private child lane: registration is dormant unless the
        // effective AA mode is temporal, and no history is begun or committed here.
        app.add_plugins(depth::TemporalDepthPlugin);
        // Reactive-mask candidates must include the app-owned SSAO pass; the graph seam is public.
        app.add_plugins(reactive::TemporalReactivePlugin);
        // Conservative current-pose metadata for later screen-space rejection of skinned models.
        app.add_plugins(skin_bounds::SkinBoundsPlugin);
        // Per-view projected model rejection rectangles; still metadata only, no history.
        app.add_plugins(rejection::ModelRejectionPlugin);
        // The fused candidate consumes depth plus both current/previous mask lanes, so it registers
        // after those producers; the graph edge below also enforces execution order.
        app.add_plugins(motion::TemporalMotionPlugin);
        // The future native resolve boundary: assembled scene color after motion, before the
        // app-owned volumetric fog/light composites that run after EndMainPass.
        app.add_plugins(scene_color::TemporalSceneColorPlugin);
        // Private opt-in Bevy-derived resolve. Cameras without the marker allocate/run nothing.
        app.add_plugins(resolve::TemporalResolvePlugin);
    }
}

/// Register added world cameras and drop removed ones. Auxiliary `Camera3d`s never carry
/// [`WorldCamera`], so they never enter temporal ownership.
fn track_world_views(
    added: Query<(Entity, Option<&TemporalTarget>), Added<WorldCamera>>,
    mut removed: RemovedComponents<WorldCamera>,
    bridge: Res<native::NativeResolveBridge>,
    mut history: ResMut<TemporalHistory>,
) {
    // Removals first: an entity re-used for a replacement view in the same frame must end up
    // registered against its new target, not dropped after the add.
    for view in removed.read() {
        history.unregister(view);
        bridge.forget(view);
    }
    for (view, target) in &added {
        history.register(view, target.copied().unwrap_or_default());
    }
}

/// A target rebuild, resize, render-scale change or DPI move carries a new [`TemporalTarget`].
#[allow(clippy::type_complexity)] // Bevy query tuples read better inline.
fn track_temporal_target(
    targets: Query<(Entity, &TemporalTarget), (With<WorldCamera>, Changed<TemporalTarget>)>,
    mut history: ResMut<TemporalHistory>,
) {
    for (view, target) in &targets {
        history.set_target(view, *target);
    }
}

/// A projection writer may mark `Projection` changed while writing the same value; compare the
/// unjittered clip matrix and invalidate only on a real change. First observation only seeds.
#[allow(clippy::type_complexity)] // Bevy query tuples read better inline.
fn track_projection(
    projections: Query<(Entity, &Projection), (With<WorldCamera>, Changed<Projection>)>,
    mut history: ResMut<TemporalHistory>,
) {
    for (view, projection) in &projections {
        if let Some(history) = history.views.get_mut(&view) {
            history.observe_projection(projection);
        }
    }
}

/// The world camera is gated off under glue screens and while the loading cover is not presented;
/// compare explicit camera values so Bevy's computed bookkeeping or a same-value write cannot
/// invalidate history.
#[allow(clippy::type_complexity)] // Bevy query tuples read better inline.
fn track_camera(
    cameras: Query<(Entity, &Camera), (With<WorldCamera>, Changed<Camera>)>,
    mut history: ResMut<TemporalHistory>,
) {
    for (view, camera) in &cameras {
        if let Some(history) = history.views.get_mut(&view) {
            history.observe_camera(camera);
        }
    }
}

/// Logging out (or a world reset) tears down the stream the history describes; a rising edge is a
/// fresh world with no representative history.
fn track_world_live(live: Res<WorldLive>, mut history: ResMut<TemporalHistory>) {
    if !live.is_changed() {
        return;
    }
    let reason = if live.0 {
        AaInvalidation::WorldTransition
    } else {
        AaInvalidation::WorldLeft
    };
    history.invalidate_all(reason);
}

/// The loading cover hides the world and then re-reveals a different one; either edge drops history.
fn track_loading_cover(
    viewer: Res<Viewer>,
    mut last_covered: Local<Option<bool>>,
    mut history: ResMut<TemporalHistory>,
) {
    let covered = viewer.world_covered;
    if last_covered.replace(covered) != Some(covered) {
        history.invalidate_all(AaInvalidation::LoadingCover);
    }
}

/// A cross-map worldport is a new scene; the same-map teleport is the app's hook.
fn track_map_changes(mut changes: MessageReader<MapChange>, mut history: ResMut<TemporalHistory>) {
    if changes.read().next().is_some() {
        history.invalidate_all(AaInvalidation::WorldTransition);
    }
}

/// Resolve the request against the camera's actual `Msaa` (the latch means `MsaaSetting` may hold a
/// pending restart value) and invalidate history only when the effective mode changes.
fn resolve_policy(
    msaa_setting: Res<MsaaSetting>,
    readiness: Res<AaReadiness>,
    cameras: Query<&Msaa, With<WorldCamera>>,
    mut policy: ResMut<AaPolicy>,
    mut history: ResMut<TemporalHistory>,
) {
    let actual_samples = cameras
        .iter()
        .next()
        .map_or(msaa_setting.samples, |msaa| msaa.samples());
    let (effective, reason) = AaPolicy::resolve(policy.requested, actual_samples, *readiness);
    if policy.effective == effective && policy.reason == reason {
        return;
    }
    let mode_changed = policy.effective != effective;
    policy.effective = effective;
    policy.reason = reason;
    if mode_changed {
        history.invalidate_all(AaInvalidation::AaModeChanged);
    }
}

#[cfg(test)]
mod tests;
