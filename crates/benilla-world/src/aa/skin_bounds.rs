//! Conservative current-pose bounds for skinned model meshes.
//!
//! This is metadata for a later screen-space rejection stage, not object motion, projected
//! rectangles, foliage coverage, TAA reconstruction or history acceptance. A `NotSkinned` result
//! is not a claim that a mesh is rigid, and a finite bound is not proof that a vertex was not
//! missed: only the mesh's actual skin attributes can make the skinned result.
//!
//! Bound construction is vertex-independent after the mesh changes: a cached envelope stores a
//! local AABB per bone. Each frame only the envelope corners are transformed by the current palette
//! rows, and the weight sums are applied conservatively. No per-vertex skinning runs every frame.
//!
//! The rig origin stays separate from the small rig-relative bounds. Missing, malformed or invalid
//! palette data is reported as `Unsafe`, never as a safe/rigid mesh.

use std::collections::{BTreeMap, BTreeSet};

use benilla_assets::materials::WowModelMaterial;
use benilla_assets::{ATTRIBUTE_WOW_JOINT_INDEX, ATTRIBUTE_WOW_JOINT_WEIGHT};
use bevy::asset::{AssetEvent, AssetId};
use bevy::camera::visibility::ViewVisibility;
use bevy::math::DVec3;
use bevy::mesh::{Mesh, MeshTag, VertexAttributeValues};
use bevy::platform::collections::{HashMap, HashSet};
use bevy::prelude::*;

use super::{resolve_policy, AaPolicy};
use crate::mesh_tag::rig_of;
use crate::rig_palette::RigPalettes;
use crate::view::WorldCamera;

/// Convert a padded f64 lower endpoint to f32 and move one representable step outward.
fn outward_down(value: f64) -> Option<f32> {
    let value = value as f32;
    value.is_finite().then(|| value.next_down())
}

/// Convert a padded f64 upper endpoint to f32 and move one representable step outward.
fn outward_up(value: f64) -> Option<f32> {
    let value = value as f32;
    value.is_finite().then(|| value.next_up())
}

/// A conservative axis-aligned bound in rig-relative space.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) struct Bounds3 {
    pub(super) min: Vec3,
    pub(super) max: Vec3,
}

impl Bounds3 {
    fn empty() -> Self {
        Self {
            min: Vec3::splat(f32::INFINITY),
            max: Vec3::splat(f32::NEG_INFINITY),
        }
    }

    fn zero() -> Self {
        Self {
            min: Vec3::ZERO,
            max: Vec3::ZERO,
        }
    }

    fn expand(&mut self, point: Vec3) {
        self.min = self.min.min(point);
        self.max = self.max.max(point);
    }

    fn is_finite(&self) -> bool {
        self.min.is_finite() && self.max.is_finite()
    }

    #[cfg(test)]
    fn contains(&self, point: Vec3, epsilon: f32) -> bool {
        point.x >= self.min.x - epsilon
            && point.x <= self.max.x + epsilon
            && point.y >= self.min.y - epsilon
            && point.y <= self.max.y + epsilon
            && point.z >= self.min.z - epsilon
            && point.z <= self.max.z + epsilon
    }
}

/// A mesh-local AABB for one bone, built once from the mesh's actual skin attributes.
#[derive(Clone, Copy, Debug, PartialEq)]
struct BoneEnvelope {
    bone: u16,
    bounds: Bounds3,
}

/// The cached, mesh-specific envelope. `weight_sum_min/max` are the original four weights; they
/// are never renormalized.
struct SkinEnvelope {
    per_bone: Vec<BoneEnvelope>,
    referenced_bones: Vec<u16>,
    weight_sum_min: f64,
    weight_sum_max: f64,
}

/// Typed reasons a mesh cannot produce a trustworthy current-pose bound.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum SkinBoundsError {
    MissingPositions,
    UnsupportedPositions,
    MissingJointIndices,
    UnsupportedJointIndices,
    MissingJointWeights,
    UnsupportedJointWeights,
    EmptyMesh,
    AttributeLengthMismatch,
    NonFinitePosition,
    NonFiniteWeight,
    NegativeWeight,
    NonFiniteWeightSum,
    InvalidPalette,
    NonFiniteRigOrigin,
    MissingRigSlot,
    MissingTag,
    MissingPalette,
    MissingOrigin,
    MissingMesh,
    MalformedEnvelope,
    NonFiniteBounds,
}

/// The component result. `NotSkinned` is deliberately not a temporal-confidence claim.
#[derive(Clone, Debug, PartialEq)]
pub(super) enum SkinBoundsResult {
    Skinned {
        rig_origin: Vec3,
        relative_bounds: Bounds3,
    },
    NotSkinned,
    Unsafe(SkinBoundsError),
}

#[derive(Component, Clone, Debug, PartialEq)]
pub(super) struct TemporalSkinBounds {
    pub(super) mesh: AssetId<Mesh>,
    pub(super) slot: u16,
    pub(super) result: SkinBoundsResult,
}

impl SkinEnvelope {
    fn from_mesh(mesh: &Mesh) -> Result<Self, SkinBoundsError> {
        let positions = match mesh.attribute(Mesh::ATTRIBUTE_POSITION) {
            Some(VertexAttributeValues::Float32x3(positions)) => positions,
            Some(_) => return Err(SkinBoundsError::UnsupportedPositions),
            None => return Err(SkinBoundsError::MissingPositions),
        };
        let joints = match mesh.attribute(ATTRIBUTE_WOW_JOINT_INDEX) {
            Some(VertexAttributeValues::Uint16x4(joints)) => joints,
            Some(_) => return Err(SkinBoundsError::UnsupportedJointIndices),
            None => return Err(SkinBoundsError::MissingJointIndices),
        };
        let weights = match mesh.attribute(ATTRIBUTE_WOW_JOINT_WEIGHT) {
            Some(VertexAttributeValues::Float32x4(weights)) => weights,
            Some(_) => return Err(SkinBoundsError::UnsupportedJointWeights),
            None => return Err(SkinBoundsError::MissingJointWeights),
        };
        if positions.len() != joints.len() || positions.len() != weights.len() {
            return Err(SkinBoundsError::AttributeLengthMismatch);
        }
        if positions.is_empty() {
            return Err(SkinBoundsError::EmptyMesh);
        }

        let mut by_bone: BTreeMap<u16, Bounds3> = BTreeMap::new();
        let mut referenced = BTreeSet::new();
        let mut weight_sum_min = f64::INFINITY;
        let mut weight_sum_max = f64::NEG_INFINITY;

        for i in 0..positions.len() {
            let position = Vec3::from_array(positions[i]);
            if !position.is_finite() {
                return Err(SkinBoundsError::NonFinitePosition);
            }
            let mut sum = 0.0f64;
            for component in weights[i] {
                if !component.is_finite() {
                    return Err(SkinBoundsError::NonFiniteWeight);
                }
                if component < 0.0 {
                    return Err(SkinBoundsError::NegativeWeight);
                }
                sum += component as f64;
            }
            if !sum.is_finite() {
                return Err(SkinBoundsError::NonFiniteWeightSum);
            }
            weight_sum_min = weight_sum_min.min(sum);
            weight_sum_max = weight_sum_max.max(sum);

            for k in 0..4 {
                let bone = joints[i][k];
                // Zero-weight entries are referenced too: a malformed or missing palette row is
                // still unsafe, even though this vertex does not consume it.
                referenced.insert(bone);
                if weights[i][k] > 0.0 {
                    by_bone
                        .entry(bone)
                        .or_insert_with(Bounds3::empty)
                        .expand(position);
                }
            }
        }

        Ok(Self {
            per_bone: by_bone
                .into_iter()
                .map(|(bone, bounds)| BoneEnvelope { bone, bounds })
                .collect(),
            referenced_bones: referenced.into_iter().collect(),
            weight_sum_min,
            weight_sum_max,
        })
    }

    /// Bound the actual shader position for this envelope. The shader computes
    /// `sum(weight_i * matrix_i * position)` without renormalizing. For a vertex with weight sum
    /// `S`, the normalized weighted point is in the convex hull of the per-bone transformed
    /// envelopes; the actual point is `S` times that point. The global [min,max] S interval is
    /// applied to every coordinate endpoint, and zero-sum vertices contribute the rig-relative
    /// origin.
    fn posed_bounds(
        &self,
        mut palette: impl FnMut(u16) -> Option<Mat4>,
    ) -> Result<Bounds3, SkinBoundsError> {
        let mut transformed: Option<(DVec3, DVec3)> = None;
        let mut per_index = 0usize;
        let mut max_p_axis = [0.0f64; 3];
        let mut max_coefficient = 0.0f64;
        let mut max_corner_extent = 0.0f64;

        for &bone in &self.referenced_bones {
            let matrix = palette(bone).ok_or(SkinBoundsError::InvalidPalette)?;
            if !matrix.is_finite() || !matrix.row(3).abs_diff_eq(Vec4::W, 1.0e-6) {
                return Err(SkinBoundsError::InvalidPalette);
            }
            for row in 0..4 {
                for col in 0..4 {
                    max_coefficient = max_coefficient.max((matrix.row(row)[col] as f64).abs());
                }
            }
            if per_index < self.per_bone.len() && self.per_bone[per_index].bone == bone {
                let bounds = self.per_bone[per_index].bounds;
                if !bounds.is_finite() || bounds.min.x > bounds.max.x {
                    return Err(SkinBoundsError::MalformedEnvelope);
                }
                let x = [bounds.min.x as f64, bounds.max.x as f64];
                let y = [bounds.min.y as f64, bounds.max.y as f64];
                let z = [bounds.min.z as f64, bounds.max.z as f64];
                for x in x {
                    for y in y {
                        for z in z {
                            // Guard against rounding in the ORIGINAL affine products before
                            // cancellation can hide their magnitude.
                            for (axis, axis_max) in max_p_axis.iter_mut().enumerate() {
                                let row = matrix.row(axis);
                                let p_axis = (row.x as f64 * x).abs()
                                    + (row.y as f64 * y).abs()
                                    + (row.z as f64 * z).abs()
                                    + (row.w as f64).abs();
                                if !p_axis.is_finite() {
                                    return Err(SkinBoundsError::NonFiniteBounds);
                                }
                                *axis_max = axis_max.max(p_axis);
                            }
                            max_corner_extent =
                                max_corner_extent.max(x.abs() + y.abs() + z.abs() + 1.0);
                            let point = matrix.as_dmat4().transform_point3(DVec3::new(x, y, z));
                            if !point.is_finite() {
                                return Err(SkinBoundsError::NonFiniteBounds);
                            }
                            transformed = Some(match transformed {
                                Some((min, max)) => (min.min(point), max.max(point)),
                                None => (point, point),
                            });
                        }
                    }
                }
                per_index += 1;
            }
        }
        if per_index != self.per_bone.len() {
            return Err(SkinBoundsError::MalformedEnvelope);
        }

        let Some((lo, hi)) = transformed else {
            // Every vertex has zero weights: the shader collapses them to the rig-relative origin.
            return Ok(Bounds3::zero());
        };
        if self.weight_sum_max <= 0.0 {
            return Ok(Bounds3::zero());
        }

        // The shader first accumulates weighted palette rows, then multiplies by the vertex.
        // Bound that intermediate separately from the final (possibly cancelled) position.
        let weighted_coefficient = max_coefficient * self.weight_sum_max;
        let weighted_product = weighted_coefficient * max_corner_extent.max(1.0);
        let f32_intermediate_limit = f32::MAX as f64 / 4.0;
        if !weighted_coefficient.is_finite()
            || !weighted_product.is_finite()
            || weighted_coefficient > f32_intermediate_limit
            || weighted_product > f32_intermediate_limit
            || max_p_axis.iter().any(|p| *p > f32_intermediate_limit)
        {
            return Err(SkinBoundsError::NonFiniteBounds);
        }

        let mut min = DVec3::splat(f64::INFINITY);
        let mut max = DVec3::splat(f64::NEG_INFINITY);
        for axis in 0..3 {
            let a = lo[axis];
            let b = hi[axis];
            let products = [
                a * self.weight_sum_min,
                a * self.weight_sum_max,
                b * self.weight_sum_min,
                b * self.weight_sum_max,
            ];
            for product in products {
                if !product.is_finite() {
                    return Err(SkinBoundsError::NonFiniteBounds);
                }
                min[axis] = min[axis].min(product);
                max[axis] = max[axis].max(product);
            }
            if self.weight_sum_min <= 0.0 {
                min[axis] = min[axis].min(0.0);
                max[axis] = max[axis].max(0.0);
            }
        }

        let mut min_f32 = [0.0f32; 3];
        let mut max_f32 = [0.0f32; 3];
        for axis in 0..3 {
            let guard = (1.0e-5_f64).max(
                128.0
                    * f32::EPSILON as f64
                    * max_p_axis[axis].max(1.0)
                    * self.weight_sum_max.max(1.0),
            );
            let Some(axis_min) = outward_down(min[axis] - guard) else {
                return Err(SkinBoundsError::NonFiniteBounds);
            };
            let Some(axis_max) = outward_up(max[axis] + guard) else {
                return Err(SkinBoundsError::NonFiniteBounds);
            };
            min_f32[axis] = axis_min;
            max_f32[axis] = axis_max;
        }
        let bounds = Bounds3 {
            min: Vec3::from_array(min_f32),
            max: Vec3::from_array(max_f32),
        };
        if !bounds.is_finite()
            || bounds.min.x > bounds.max.x
            || bounds.min.y > bounds.max.y
            || bounds.min.z > bounds.max.z
        {
            return Err(SkinBoundsError::NonFiniteBounds);
        }
        Ok(bounds)
    }
}

#[derive(Resource, Default)]
struct SkinEnvelopeCache {
    entries: HashMap<AssetId<Mesh>, Result<SkinEnvelope, SkinBoundsError>>,
    builds: u64,
}

impl SkinEnvelopeCache {
    fn envelope(
        &mut self,
        id: AssetId<Mesh>,
        mesh: &Mesh,
    ) -> Result<&SkinEnvelope, SkinBoundsError> {
        if !self.entries.contains_key(&id) {
            self.builds += 1;
            self.entries.insert(id, SkinEnvelope::from_mesh(mesh));
        }
        match self.entries.get(&id) {
            Some(Ok(envelope)) => Ok(envelope),
            Some(Err(error)) => Err(*error),
            None => unreachable!("entry was just inserted"),
        }
    }

    fn evict(&mut self, id: AssetId<Mesh>) {
        self.entries.remove(&id);
    }

    fn retain_referenced(&mut self, referenced: &HashSet<AssetId<Mesh>>) {
        self.entries.retain(|id, _| referenced.contains(id));
    }

    fn clear(&mut self) {
        self.entries.clear();
    }

    #[cfg(test)]
    fn builds(&self) -> u64 {
        self.builds
    }

    #[cfg(test)]
    fn contains(&self, id: AssetId<Mesh>) -> bool {
        self.entries.contains_key(&id)
    }
}

/// Resolve one mesh's current-pose metadata through the production cache/palette path.
fn skinned_result(
    id: AssetId<Mesh>,
    mesh: &Mesh,
    slot: u16,
    palettes: Option<&RigPalettes>,
    cache: &mut SkinEnvelopeCache,
) -> SkinBoundsResult {
    let has_joints = mesh.contains_attribute(ATTRIBUTE_WOW_JOINT_INDEX);
    let has_weights = mesh.contains_attribute(ATTRIBUTE_WOW_JOINT_WEIGHT);
    if !has_joints && !has_weights {
        return SkinBoundsResult::NotSkinned;
    }
    if slot == 0 {
        return SkinBoundsResult::Unsafe(SkinBoundsError::MissingRigSlot);
    }
    let Some(palettes) = palettes else {
        return SkinBoundsResult::Unsafe(SkinBoundsError::MissingPalette);
    };
    let envelope = match cache.envelope(id, mesh) {
        Ok(envelope) => envelope,
        Err(error) => return SkinBoundsResult::Unsafe(error),
    };
    let Some(rig_origin) = palettes.slot_origin(slot) else {
        return SkinBoundsResult::Unsafe(SkinBoundsError::MissingOrigin);
    };
    if !rig_origin.is_finite() {
        return SkinBoundsResult::Unsafe(SkinBoundsError::NonFiniteRigOrigin);
    }
    match envelope.posed_bounds(|bone| palettes.rig_row(slot, bone)) {
        Ok(relative_bounds) => SkinBoundsResult::Skinned {
            rig_origin,
            relative_bounds,
        },
        Err(error) => SkinBoundsResult::Unsafe(error),
    }
}

fn should_prepare(policy: Option<&AaPolicy>, cameras: &Query<&Camera, With<WorldCamera>>) -> bool {
    policy.is_some_and(|policy| policy.effective().is_temporal())
        && cameras.iter().any(|camera| camera.is_active)
}

/// Prepare current-pose metadata only while effective temporal AA has an active world view.
///
/// `ViewVisibility::get()` is used only to retire an explicitly hidden entity. It is not proof
/// that the entity belongs to the world camera; later per-view rejection owns that classification.
#[allow(clippy::type_complexity)] // Bevy query tuples read better inline.
fn prepare_skin_bounds(
    mut commands: Commands,
    policy: Option<Res<AaPolicy>>,
    cameras: Query<&Camera, With<WorldCamera>>,
    mut cache: ResMut<SkinEnvelopeCache>,
    meshes: Option<Res<Assets<Mesh>>>,
    palettes: Option<Res<RigPalettes>>,
    models: Query<
        (
            Entity,
            &Mesh3d,
            Option<&MeshTag>,
            Option<&ViewVisibility>,
            Option<&TemporalSkinBounds>,
        ),
        With<MeshMaterial3d<WowModelMaterial>>,
    >,
    mut events: Option<MessageReader<AssetEvent<Mesh>>>,
    transient: Query<Entity, With<TemporalSkinBounds>>,
) {
    if !should_prepare(policy.as_deref(), &cameras) {
        if let Some(events) = events.as_mut() {
            for _ in events.read() {}
        }
        for entity in &transient {
            commands.entity(entity).remove::<TemporalSkinBounds>();
        }
        cache.clear();
        return;
    }

    if let Some(events) = events.as_mut() {
        for event in events.read() {
            if let AssetEvent::Modified { id } | AssetEvent::Removed { id } = event {
                cache.evict(*id);
            }
        }
    }

    let Some(meshes) = meshes else {
        for entity in &transient {
            commands.entity(entity).remove::<TemporalSkinBounds>();
        }
        cache.clear();
        return;
    };

    let mut referenced = HashSet::new();
    let mut seen = HashSet::new();
    for (entity, mesh3d, tag, visibility, existing) in &models {
        if visibility.is_some_and(|visibility| !visibility.get()) {
            commands.entity(entity).remove::<TemporalSkinBounds>();
            continue;
        }
        seen.insert(entity);
        let id = mesh3d.0.id();
        referenced.insert(id);
        let slot = tag.map(|tag| rig_of(tag.0)).unwrap_or(0);
        let result = if tag.is_none() {
            SkinBoundsResult::Unsafe(SkinBoundsError::MissingTag)
        } else {
            match meshes.get(id) {
                Some(mesh) => skinned_result(id, mesh, slot, palettes.as_deref(), &mut cache),
                None => SkinBoundsResult::Unsafe(SkinBoundsError::MissingMesh),
            }
        };
        let next = TemporalSkinBounds {
            mesh: id,
            slot,
            result,
        };
        if existing != Some(&next) {
            commands.entity(entity).insert(next);
        }
    }

    for entity in &transient {
        if !seen.contains(&entity) {
            commands.entity(entity).remove::<TemporalSkinBounds>();
        }
    }
    cache.retain_referenced(&referenced);
}

/// Private child plugin: registered from [`super::AaPlugin`].
pub(super) struct SkinBoundsPlugin;

impl Plugin for SkinBoundsPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<SkinEnvelopeCache>()
            .add_systems(Last, prepare_skin_bounds.after(resolve_policy));
    }
}

#[cfg(test)]
mod tests;
