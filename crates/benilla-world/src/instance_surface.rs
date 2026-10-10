// MONKEY: generic per-rig surface rows; zero leaves the stock model shader unchanged.
use std::sync::Arc;

use bevy::prelude::*;
use bevy::render::extract_resource::{ExtractResource, ExtractResourcePlugin};
use bevy::render::renderer::RenderQueue;
use bevy::render::{Render, RenderApp, RenderSystems};

use crate::mesh_tag::MAX_RIG_SLOTS;
use crate::rig_palette::RigPaletteMirrors;

// MONKEY: one vec4 per slot, after water clips and before the runtime palette tail.
pub(crate) fn region_offset() -> u64 {
    crate::straddle::region_offset() + crate::straddle::region_bytes()
}

pub(crate) fn region_bytes() -> u64 {
    MAX_RIG_SLOTS as u64 * 16
}

/// MONKEY: `(whole_body_wet, lower_body_wet, waterline_bevy_y, mask_feather_y)`.
/// Wetness is clamped to 0..1, feather to at least 0.01 yd. Invalid/nonfinite rows clear the slot.
/// Slot 0 is the no-rig sentinel. The resource is shared with the render world on extraction.
#[derive(Resource, Clone, ExtractResource)]
pub struct InstanceSurfaceParams {
    slots: Arc<Vec<[f32; 4]>>,
    generation: u64,
}

impl Default for InstanceSurfaceParams {
    fn default() -> Self {
        Self {
            slots: Arc::new(vec![[0.0; 4]; MAX_RIG_SLOTS]),
            generation: 0,
        }
    }
}

impl InstanceSurfaceParams {
    /// MONKEY: set a live rig slot's surface row; invalid slots are ignored.
    pub fn set(&mut self, slot: u16, row: Vec4) {
        let i = slot as usize;
        if i == 0 || i >= MAX_RIG_SLOTS {
            return;
        }
        // MONKEY: reject finite extremes that cannot form distinct, finite smoothstep edges.
        let feather = row.w.max(0.01);
        let (lo, hi) = (row.z - feather, row.z + feather);
        let row = if !row.is_finite()
            || (row.x <= 0.0 && row.y <= 0.0)
            || !lo.is_finite()
            || !hi.is_finite()
            || !(hi - lo).is_finite()
            || lo >= hi
        {
            [0.0; 4]
        } else {
            [row.x.clamp(0.0, 1.0), row.y.clamp(0.0, 1.0), row.z, feather]
        };
        if self.slots[i] != row {
            Arc::make_mut(&mut self.slots)[i] = row;
            self.generation = self.generation.wrapping_add(1);
        }
    }

    /// MONKEY: clear on rig removal as well as when the caller disables its surface effect.
    pub fn clear(&mut self, slot: u16) {
        self.set(slot, Vec4::ZERO);
    }

    /// MONKEY: read the sanitized row; invalid slots return the inert row.
    pub fn get(&self, slot: u16) -> Vec4 {
        Vec4::from_array(self.slots.get(slot as usize).copied().unwrap_or([0.0; 4]))
    }
}

// MONKEY: generation-gated upload to the world, glue booth and portrait/pane buffers.
fn upload_instance_surface(
    queue: Res<RenderQueue>,
    shared: Option<Res<crate::lighting::SharedLightBuffer>>,
    mirrors: Option<Res<RigPaletteMirrors>>,
    surface: Res<InstanceSurfaceParams>,
    mut last: Local<Vec<(bevy::render::render_resource::BufferId, u64)>>,
) {
    let targets: Vec<_> = shared
        .iter()
        .map(|s| &s.0)
        .chain(mirrors.iter().flat_map(|m| m.0.values()))
        .collect();
    last.retain(|(id, _)| targets.iter().any(|b| b.id() == *id));
    for buffer in targets {
        let stamp = (buffer.id(), surface.generation);
        if last.contains(&stamp) {
            continue;
        }
        queue.write_buffer(
            buffer,
            region_offset(),
            bytemuck::cast_slice(surface.slots.as_slice()),
        );
        last.retain(|(id, _)| *id != buffer.id());
        last.push(stamp);
    }
}

// MONKEY: register the independent renderer seam alongside the other slot tables.
pub(crate) fn plugin(app: &mut App) {
    app.init_resource::<InstanceSurfaceParams>()
        .add_plugins(ExtractResourcePlugin::<InstanceSurfaceParams>::default());
    if let Some(render) = app.get_sub_app_mut(RenderApp) {
        render.add_systems(
            Render,
            upload_instance_surface.in_set(RenderSystems::PrepareResources),
        );
    }
}

// MONKEY: pin the zero path, bounds, sanitization and slot recycling API.
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zero_is_inert_and_invalid_slots_are_ignored() {
        let mut surface = InstanceSurfaceParams::default();
        surface.set(7, Vec4::ZERO);
        surface.clear(7);
        surface.set(0, Vec4::ONE);
        surface.set(u16::MAX, Vec4::ONE);
        assert_eq!(surface.generation, 0);
        assert!(surface.slots.iter().all(|r| *r == [0.0; 4]));
        assert_eq!(surface.get(u16::MAX), Vec4::ZERO);
    }

    // MONKEY: the Rust region order must match `WowLight`'s declaration order in the shader:
    // the straddle clips, then the vec4 surface rows (16-byte aligned), then the runtime palettes.
    #[test]
    fn the_shader_declares_the_surface_rows_between_clips_and_palettes() {
        let src = include_str!("../../benilla-assets/src/shaders/wow_model.wgsl");
        let clip = src
            .find("water_clip: array<vec2<f32>, 2048>,")
            .expect("clip table");
        let surface = src
            .find("rig_surface: array<vec4<f32>, 2048>,")
            .expect("surface table");
        let palettes = src
            .find("palettes: array<vec4<f32>>,")
            .expect("palette tail");
        assert!(clip < surface && surface < palettes);
        assert_eq!(MAX_RIG_SLOTS, 2048);
        assert_eq!(region_offset() % 16, 0, "vec4 rows need 16-byte alignment");
        assert_eq!(crate::straddle::region_offset() % 16, 0);
    }

    #[test]
    fn set_clear_and_extract_preserve_independent_slots() {
        let mut surface = InstanceSurfaceParams::default();
        let row = Vec4::new(0.5, 1.0, 12.0, 0.2);
        surface.set(7, row);
        let extracted = surface.clone();
        surface.set(8, row);
        surface.set(7, row);
        assert_eq!(surface.generation, 2);
        surface.clear(7);
        assert_eq!(surface.get(7), Vec4::ZERO);
        assert_eq!(surface.get(8), row);
        assert_eq!(extracted.get(7), row);
        surface.clear(7);
        assert_eq!(surface.generation, 3);
    }

    #[test]
    fn bad_rows_cannot_upload_nan_or_unbounded_wetness() {
        let mut surface = InstanceSurfaceParams::default();
        surface.set(7, Vec4::new(2.0, -1.0, 12.0, -0.1));
        assert_eq!(surface.get(7), Vec4::new(1.0, 0.0, 12.0, 0.01));
        surface.set(7, Vec4::new(1.0, 1.0, f32::NAN, 1.0));
        assert_eq!(surface.get(7), Vec4::ZERO);
        surface.set(7, Vec4::new(1.0, 1.0, 0.0, f32::MAX));
        assert_eq!(surface.get(7), Vec4::ZERO);
        surface.set(7, Vec4::new(1.0, 1.0, f32::MAX, 0.01));
        assert_eq!(surface.get(7), Vec4::ZERO);
    }

    // MONKEY: removing or replacing RigSkin clears the slot before allocator reuse.
    #[test]
    fn rig_removal_clears_surface_before_slot_reuse() {
        use crate::rig_palette::{RigPalettes, RigSkin};
        let mut world = World::new();
        world.init_resource::<RigPalettes>();
        world.init_resource::<InstanceSurfaceParams>();
        let skin = RigSkin::allocate_bones(
            &mut world.resource_mut::<RigPalettes>(),
            1,
            Handle::default(),
        )
        .unwrap();
        let slot = skin.slot;
        world
            .resource_mut::<InstanceSurfaceParams>()
            .set(slot, Vec4::ONE);
        let entity = world.spawn(skin).id();
        world.entity_mut(entity).remove::<RigSkin>();
        assert_eq!(
            world.resource::<InstanceSurfaceParams>().get(slot),
            Vec4::ZERO
        );
        let replacement = RigSkin::allocate_bones(
            &mut world.resource_mut::<RigPalettes>(),
            1,
            Handle::default(),
        )
        .unwrap();
        assert_eq!(replacement.slot, slot);
        world.entity_mut(entity).insert(replacement);
        world
            .resource_mut::<InstanceSurfaceParams>()
            .set(slot, Vec4::ONE);
        let next = RigSkin::allocate_bones(
            &mut world.resource_mut::<RigPalettes>(),
            1,
            Handle::default(),
        )
        .unwrap();
        world.entity_mut(entity).insert(next);
        assert_eq!(
            world.resource::<InstanceSurfaceParams>().get(slot),
            Vec4::ZERO
        );
    }
}
