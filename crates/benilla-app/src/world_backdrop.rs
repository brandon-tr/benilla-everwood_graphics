//! The world as the UI's backdrop: the world's final pass, the FFXGlow combine ([`FfxBackdrop`]),
//! is the first draw of the UI camera's main pass, straight into the UI's byte target.
//!
//! The reference's fixed-function device blends in gamma bytes. The world lane emits gamma bytes,
//! and the UI lane blends in an `Rgba8UnormSrgb` target, so every UI blend is arithmetic on the
//! gamma value (`alphaMode="ADD"` is `dst + texel·α`, clamped, as EGxBlend 3). Drawing the world
//! inside that target keeps the UI-over-world blend in bytes too; composited through the sRGB
//! swapchain it would run in linear and lighten every translucent UI pixel over the world (a docked
//! chat tab, black at α 102/255, must leave the scene at 60 %; a linear blend leaves 77.5 %).
//!
//! The world camera's own target is a size-carrier: a one-byte image at the world's render size
//! that nothing writes, kept because Bevy sizes a view's main texture from its target and the
//! camera's logical viewport rides it ([`retarget_world_camera`]).
//!
//! [`RenderScale`] renders the world at `window × scale` while the UI stays native. At exactly 2.0
//! the combine's bilinear read is a 2×2 box average, so supersampling needs no filter of its own.
//! The camera's logical viewport must not move with it ([`render_target_for`]).

use bevy::asset::RenderAssetUsages;
use bevy::camera::RenderTarget;
use bevy::prelude::*;
use bevy::render::camera::MipBias;
use bevy::render::render_resource::{Extent3d, TextureDimension, TextureFormat, TextureUsages};
use bevy::window::PrimaryWindow;

use crate::ui_pass::PlayerUiCamera;
use benilla_world::aa::TemporalTarget;
use benilla_world::ffx_glow::FfxBackdrop;
use benilla_world::view::WorldCamera;

/// The render scale, the `renderScale` CVar, not a 1.12 CVar: it scales the 3D and keeps the
/// interface native. At the default 1.0 [`render_target_for`] returns the window's own numbers.
#[derive(Resource, Clone, Copy, PartialEq, Debug)]
pub(crate) struct RenderScale(pub(crate) f32);

/// The settable range of [`RenderScale`], shared by the CVar and `$WOW_RENDER_SCALE`. Past 2 on
/// purpose: supersampling prices a pixel on a machine whose present is railed at vsync.
pub(crate) const RENDER_SCALE_RANGE: std::ops::RangeInclusive<f32> = 0.25..=4.0;

/// The per-axis ceiling in physical px, `wgpu::Limits::default()`'s `max_texture_dimension_2d`,
/// applied to the ratio ([`render_target_for`]) so the picture shrinks rather than reshapes.
const MAX_RENDER_AXIS: u32 = 8192;

impl Default for RenderScale {
    /// `$WOW_RENDER_SCALE` overrides the default for the session only, never into `config.toml`.
    fn default() -> Self {
        let scale = std::env::var("WOW_RENDER_SCALE")
            .ok()
            .and_then(|v| v.parse::<f32>().ok())
            .filter(|v| v.is_finite())
            .map_or(1.0, |v| {
                v.clamp(*RENDER_SCALE_RANGE.start(), *RENDER_SCALE_RANGE.end())
            });
        Self(scale)
    }
}

/// The backdrop's size and the world camera's target scale factor, for a window of `px` physical
/// pixels at `window_factor`, rendered at `scale`.
///
/// 1. The ratio, never one axis, clamps to [`MAX_RENDER_AXIS`], so the image keeps the window's
///    aspect.
/// 2. `size.x / factor`, the camera's logical viewport width, is the window's logical width at any
///    `scale`: the factor derives from the size built, after the `round()`, since every pick ray is
///    denominated in it.
///
/// At `scale == 1.0` both are exact in IEEE, so the window's own numbers come back bit for bit.
fn render_target_for(px: UVec2, window_factor: f32, scale: f32) -> (UVec2, f32) {
    let px = px.max(UVec2::ONE);
    let ceiling = |axis: u32| MAX_RENDER_AXIS as f32 / axis as f32;
    let scale = scale
        .clamp(*RENDER_SCALE_RANGE.start(), *RENDER_SCALE_RANGE.end())
        .min(ceiling(px.x))
        .min(ceiling(px.y));
    let size = (px.as_vec2() * scale)
        .round()
        .as_uvec2()
        .clamp(UVec2::ONE, UVec2::splat(MAX_RENDER_AXIS));
    (size, window_factor * size.x as f32 / px.x as f32)
}

/// The texture LOD bias a render at `effective` scale owes its mipmapped textures: `log2(scale)`,
/// clamped at 0 so supersampling keeps its sharper mips.
///
/// WebGPU samplers carry no LOD bias, so it rides [`MipBias`] into the view uniform, which the PBR
/// lane applies and our own shaders must too (`terrain.wgsl`, `static_gx.wgsl`, `liquid.wgsl`, the
/// coverage re-sample in `wow_model.wgsl`).
fn mip_bias(effective: f32) -> f32 {
    effective.log2().min(0.0)
}

/// The world camera's target, the size-carrier, in physical px × [`RenderScale`]; at 1.0 the
/// combine's read of the world is an identity resample.
#[derive(Resource)]
pub(crate) struct WorldBackdrop {
    /// The image the world camera targets. Nothing writes or samples it (the camera's final pass
    /// is the UI camera's, [`FfxBackdrop`]); it gives the view its size and logical viewport.
    pub(crate) image: Handle<Image>,
    /// The size the image was last built at, in physical px: the resize gate.
    size: UVec2,
    /// The image and scale factor the camera was last stamped with. Both gate the re-stamp: a
    /// rebuilt backdrop is a new asset ([`track_render_size`]). `None` until the first stamp.
    stamped: Option<(AssetId<Image>, f32)>,
}

impl WorldBackdrop {
    /// The world's render size in physical px, after the rounding and the axis ceiling.
    pub(crate) fn render_size(&self) -> UVec2 {
        self.size
    }
}

/// A fresh size-carrier at `size` physical px: one byte a pixel, a render attachment only, its CPU
/// copy dropped once the render world has it.
fn new_world_target(size: UVec2) -> Image {
    let mut image = Image::new_fill(
        Extent3d {
            width: size.x.max(1),
            height: size.y.max(1),
            depth_or_array_layers: 1,
        },
        TextureDimension::D2,
        &[0],
        TextureFormat::R8Unorm,
        RenderAssetUsages::RENDER_WORLD,
    );
    image.texture_descriptor.usage = TextureUsages::RENDER_ATTACHMENT;
    image
}

fn window_physical_size(window: &Window) -> UVec2 {
    UVec2::new(
        window.physical_width().max(1),
        window.physical_height().max(1),
    )
}

fn setup_backdrop(
    mut commands: Commands,
    mut images: ResMut<Assets<Image>>,
    windows: Query<&Window, With<PrimaryWindow>>,
    scale: Res<RenderScale>,
) {
    let (size, _) = windows.single().map_or((UVec2::new(1280, 720), 1.0), |w| {
        render_target_for(
            window_physical_size(w),
            w.resolution.scale_factor(),
            scale.0,
        )
    });
    let image = images.add(new_world_target(size));
    commands.insert_resource(WorldBackdrop {
        image,
        size,
        // Nothing stamped yet, so `retarget_world_camera`'s first run always fires.
        stamped: None,
    });
}

/// Keep the target at `window × `[`RenderScale`]; runs before the stamp, whose factor must pair
/// with this size.
///
/// A rebuild publishes a new asset and removes the old one, never writing through the old handle:
/// `AssetId<Image>` names a GPU texture, and anything keyed on it would keep the retired texture.
fn track_render_size(
    mut backdrop: ResMut<WorldBackdrop>,
    mut images: ResMut<Assets<Image>>,
    windows: Query<&Window, With<PrimaryWindow>>,
    scale: Res<RenderScale>,
) {
    let Ok(window) = windows.single() else {
        return;
    };
    let (size, _) = render_target_for(
        window_physical_size(window),
        window.resolution.scale_factor(),
        scale.0,
    );
    if size == backdrop.size {
        return;
    }
    let retired = std::mem::replace(&mut backdrop.image, images.add(new_world_target(size)));
    images.remove(&retired);
    backdrop.size = size;
    // Logged on every change: the rendered pixel count is the one term a probe cannot infer.
    info!(
        "render scale {:.3}: world renders at {}x{} into a {}x{} window",
        scale.0,
        size.x,
        size.y,
        window.physical_width(),
        window.physical_height()
    );
}

/// Point the world camera at the backdrop instead of the swapchain, carrying the window's scale
/// factor. A system, as the camera has two spawn sites; `benilla-worldview` keeps the swapchain.
///
/// `From<Handle<Image>>` stamps `scale_factor: 1.0`, which on a physical-px image makes the logical
/// viewport the physical size and throws every pick ray and `world_to_viewport` off by the display
/// scale. With the window's factor the logical viewport stays the window's, and render scale moves
/// size and factor together, so projection and picking never see it. Re-stamped when the image or
/// the factor changes, with the [`MipBias`] ([`mip_bias`]) inserted on every stamp.
fn retarget_world_camera(
    mut commands: Commands,
    mut backdrop: ResMut<WorldBackdrop>,
    windows: Query<&Window, With<PrimaryWindow>>,
    added: Query<(), Added<WorldCamera>>,
    mut cameras: Query<(Entity, &mut RenderTarget, Option<&mut TemporalTarget>), With<WorldCamera>>,
) {
    let (px, window_factor) = windows.single().map_or((UVec2::ONE, 1.0), |w| {
        (window_physical_size(w), w.resolution.scale_factor())
    });
    // The built size, not the requested one: `track_render_size` may have clamped.
    let scale_factor = window_factor * backdrop.size.x as f32 / px.x.max(1) as f32;
    // The image as well as the factor: a pure window resize rebuilds without moving the factor.
    let want = (backdrop.image.id(), scale_factor);
    let moved = backdrop.stamped != Some(want);
    if !moved && added.is_empty() {
        return;
    }
    backdrop.stamped = Some(want);
    let bias = MipBias(mip_bias(scale_factor / window_factor));
    let temporal_target = TemporalTarget::new(backdrop.size, scale_factor);
    for (entity, mut current, temporal) in &mut cameras {
        *current = RenderTarget::Image(bevy::camera::ImageRenderTarget {
            handle: backdrop.image.clone(),
            scale_factor,
        });
        commands.entity(entity).insert(bias.clone());
        // The AA history is per target configuration: a rebuilt image, a resize or a DPI move
        // lands here and invalidates that view on the next `Last` pass.
        match temporal {
            Some(mut current) => {
                if *current != temporal_target {
                    *current = temporal_target;
                }
            }
            None => {
                commands.entity(entity).insert(temporal_target);
            }
        }
    }
}

/// Point the player-UI camera's ground pass ([`FfxBackdrop`]) at the active world camera; with
/// none it is `None`, and the UI camera clears to transparent rather than show a stale world frame.
fn claim_backdrop(
    cameras: Query<(Entity, &Camera), With<WorldCamera>>,
    mut ui: Query<&mut FfxBackdrop, With<PlayerUiCamera>>,
) {
    let source = cameras
        .iter()
        .find_map(|(entity, camera)| camera.is_active.then_some(entity));
    for mut backdrop in &mut ui {
        // Compare first: a write would trip change detection for nothing.
        if backdrop.source != source {
            backdrop.source = source;
        }
    }
}

/// Owns the world camera's target (the size-carrier) and the UI camera's claim on the world.
pub(crate) struct WorldBackdropPlugin;

/// Render scale's change callback, clamped to [`RENDER_SCALE_RANGE`]; the backdrop and the
/// camera's factor follow together on the next frame.
pub(crate) fn on_cvar(
    ev: &crate::cvars::CvarChanged,
    mut scale: bevy::prelude::Mut<'_, RenderScale>,
) {
    if ev.is("renderScale") {
        scale.0 = ev
            .num()
            .clamp(*RENDER_SCALE_RANGE.start(), *RENDER_SCALE_RANGE.end());
    }
}

impl Plugin for WorldBackdropPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<RenderScale>()
            .add_systems(Startup, setup_backdrop)
            .add_systems(
                Update,
                (track_render_size, retarget_world_camera, claim_backdrop)
                    .chain()
                    // After the calls, whose `renderScale` callback writes the knob.
                    .after(crate::vm_pass::VmPass::End),
            );
    }
}

#[cfg(test)]
mod tests;
