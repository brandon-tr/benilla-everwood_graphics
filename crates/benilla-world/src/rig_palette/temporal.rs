//! Stable previous-rig GPU data for the native TAA adapter.

use bevy::render::render_resource::{BufferDescriptor, BufferUsages};
use bevy::render::renderer::RenderDevice;

use benilla_assets::materials::RigTaaBinds;
pub(crate) mod snapshot;

pub(crate) const HEADER_BYTES: u64 = 16;
pub(crate) const MONKEY_ROWS: u64 = crate::lighting::MONKEY_FRAME_ROWS as u64;
pub(crate) const MONKEY_BYTES: u64 = MONKEY_ROWS * 16;
pub(crate) const SLOT_META_OFFSET: u64 = HEADER_BYTES + MONKEY_BYTES;
pub(crate) const SLOT_META_BYTES: u64 = crate::mesh_tag::MAX_RIG_SLOTS as u64 * 16;
pub(crate) const SLOT_BASES_OFFSET: u64 = SLOT_META_OFFSET + SLOT_META_BYTES;
pub(crate) const SLOT_BASES_BYTES: u64 = crate::mesh_tag::MAX_RIG_SLOTS as u64 * 4;
pub(crate) const SLOT_ORIGINS_OFFSET: u64 = SLOT_BASES_OFFSET + SLOT_BASES_BYTES;
pub(crate) const SLOT_ORIGINS_BYTES: u64 = crate::mesh_tag::MAX_RIG_SLOTS as u64 * 16;
pub(crate) const PALETTE_OFFSET: u64 = SLOT_ORIGINS_OFFSET + SLOT_ORIGINS_BYTES;
pub(crate) const PALETTE_BONES: u64 = super::MAX_PALETTE_BONES as u64;
pub(crate) const BONE_BYTES: u64 = super::BONE_BYTES;
pub(crate) const PALETTE_BYTES: u64 = PALETTE_BONES * BONE_BYTES;
pub(crate) const PREVIOUS_RIG_BYTES: u64 = PALETTE_OFFSET + PALETTE_BYTES;

pub(crate) fn new_rig_taa_binds(device: &RenderDevice) -> RigTaaBinds {
    let previous = device.create_buffer(&BufferDescriptor {
        label: Some("rig_taa_previous"),
        size: PREVIOUS_RIG_BYTES,
        usage: BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
        mapped_at_creation: true,
    });
    previous.slice(..).get_mapped_range_mut().fill(0);
    previous.unmap();
    RigTaaBinds { previous }
}

#[cfg(test)]
mod tests;
