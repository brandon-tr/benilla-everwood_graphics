use super::*;
use crate::aa::{AaMode, AaReadiness};
use bevy::asset::uuid::Uuid;
use bevy::asset::{Assets, RenderAssetUsages};
use bevy::math::Affine3A;
use bevy::render::render_resource::PrimitiveTopology;

fn mesh_with(positions: &[[f32; 3]], joints: &[[u16; 4]], weights: &[[f32; 4]]) -> Mesh {
    let mut mesh = Mesh::new(
        PrimitiveTopology::TriangleList,
        RenderAssetUsages::default(),
    );
    mesh.insert_attribute(
        Mesh::ATTRIBUTE_POSITION,
        VertexAttributeValues::Float32x3(positions.to_vec()),
    );
    mesh.insert_attribute(
        ATTRIBUTE_WOW_JOINT_INDEX,
        VertexAttributeValues::Uint16x4(joints.to_vec()),
    );
    mesh.insert_attribute(
        ATTRIBUTE_WOW_JOINT_WEIGHT,
        VertexAttributeValues::Float32x4(weights.to_vec()),
    );
    mesh
}

fn id(n: u128) -> AssetId<Mesh> {
    AssetId::Uuid {
        uuid: Uuid::from_u128(n),
    }
}

fn palette() -> RigPalettes {
    let mut palettes = RigPalettes::default();
    let (slot, _) = palettes.alloc(2).unwrap();
    palettes.write_rider(
        slot,
        Affine3A::from_rotation_translation(Quat::from_rotation_y(0.3), Vec3::new(1.0, 2.0, 3.0)),
        Vec3::new(9000.0, 10.0, -4.0),
    );
    palettes
}

fn blended_rows(weights: [f32; 4], palette: [Mat4; 4]) -> [[f32; 4]; 4] {
    let mut blended = [[0.0f32; 4]; 4];
    for (row, out_row) in blended.iter_mut().enumerate() {
        for (col, out) in out_row.iter_mut().enumerate() {
            let mut acc = 0.0f32;
            for k in 0..4 {
                let term = std::hint::black_box(
                    weights[k] * std::hint::black_box(palette[k].row(row)[col]),
                );
                acc = std::hint::black_box(acc + term);
            }
            *out = std::hint::black_box(acc);
        }
    }
    blended
}

/// Shader-order evaluation with explicit separate f32 multiply/add barriers.
fn shader_order_skin_separate_f32(weights: [f32; 4], palette: [Mat4; 4], vertex: Vec3) -> Vec3 {
    let rows = blended_rows(weights, palette);
    let mut out = [0.0f32; 4];
    for row in 0..4 {
        let mut acc = std::hint::black_box(rows[row][0] * vertex.x);
        acc = std::hint::black_box(acc + std::hint::black_box(rows[row][1] * vertex.y));
        acc = std::hint::black_box(acc + std::hint::black_box(rows[row][2] * vertex.z));
        acc = std::hint::black_box(acc + std::hint::black_box(rows[row][3]));
        out[row] = std::hint::black_box(acc);
    }
    Vec3::new(out[0], out[1], out[2])
}

/// Shader-order evaluation with a valid fused homogeneous product.
fn shader_order_skin_fused_f32(weights: [f32; 4], palette: [Mat4; 4], vertex: Vec3) -> Vec3 {
    let rows = blended_rows(weights, palette);
    let mut out = [0.0f32; 4];
    for row in 0..4 {
        let acc = std::hint::black_box(rows[row][3].mul_add(
            1.0,
            rows[row][2].mul_add(
                vertex.z,
                rows[row][1].mul_add(vertex.y, rows[row][0].mul_add(vertex.x, 0.0)),
            ),
        ));
        out[row] = std::hint::black_box(acc);
    }
    Vec3::new(out[0], out[1], out[2])
}

#[test]
fn reported_scale_translation_cancellation_is_enclosed() {
    let x = f32::from_bits(1.0f32.to_bits() + 1);
    let mesh = mesh_with(&[[x, 0.0, 0.0]], &[[0, 0, 0, 0]], &[[1.0, 0.0, 0.0, 0.0]]);
    let envelope = SkinEnvelope::from_mesh(&mesh).unwrap();
    let matrix = Mat4::from_cols(
        Vec4::new(100_000_000.0, 0.0, 0.0, 0.0),
        Vec4::new(0.0, 1.0, 0.0, 0.0),
        Vec4::new(0.0, 0.0, 1.0, 0.0),
        Vec4::new(-100_000_000.0, 0.0, 0.0, 1.0),
    );
    let weights = [1.0, 0.0, 0.0, 0.0];
    let palette = [matrix, Mat4::IDENTITY, Mat4::IDENTITY, Mat4::IDENTITY];
    let separate = shader_order_skin_separate_f32(weights, palette, Vec3::new(x, 0.0, 0.0));
    let fused = shader_order_skin_fused_f32(weights, palette, Vec3::new(x, 0.0, 0.0));
    assert_eq!(separate.x, 8.0, "the reported separate-f32 result");

    let bounds = envelope
        .posed_bounds(|_| Some(matrix))
        .expect("cancellation pose is representable");
    assert!(
        bounds.contains(separate, 0.0),
        "separate {separate:?} must be inside {bounds:?}"
    );
    assert!(
        bounds.contains(fused, 0.0),
        "fused {fused:?} must be inside {bounds:?}"
    );
}

#[test]
fn cancellation_across_multiple_weighted_bones_is_enclosed() {
    let x = f32::from_bits(1.0f32.to_bits() + 1);
    let mesh = mesh_with(&[[x, 0.0, 0.0]], &[[0, 1, 0, 0]], &[[0.25, 0.75, 0.0, 0.0]]);
    let envelope = SkinEnvelope::from_mesh(&mesh).unwrap();
    let m0 = Mat4::from_cols(
        Vec4::new(100_000_000.0, 0.0, 0.0, 0.0),
        Vec4::new(0.0, 1.0, 0.0, 0.0),
        Vec4::new(0.0, 0.0, 1.0, 0.0),
        Vec4::new(-100_000_000.0, 0.0, 0.0, 1.0),
    );
    let m1 = Mat4::from_cols(
        Vec4::new(-100_000_000.0, 0.0, 0.0, 0.0),
        Vec4::new(0.0, 1.0, 0.0, 0.0),
        Vec4::new(0.0, 0.0, 1.0, 0.0),
        Vec4::new(100_000_000.0, 0.0, 0.0, 1.0),
    );
    let bounds = envelope
        .posed_bounds(|bone| match bone {
            0 => Some(m0),
            1 => Some(m1),
            _ => None,
        })
        .expect("multi-bone cancellation bounds");
    let palette = [m0, m1, Mat4::IDENTITY, Mat4::IDENTITY];
    let weights = [0.25, 0.75, 0.0, 0.0];
    let separate = shader_order_skin_separate_f32(weights, palette, Vec3::new(x, 0.0, 0.0));
    let fused = shader_order_skin_fused_f32(weights, palette, Vec3::new(x, 0.0, 0.0));
    assert!(
        bounds.contains(separate, 0.0),
        "multi-bone separate {separate:?} not in {bounds:?}"
    );
    assert!(
        bounds.contains(fused, 0.0),
        "multi-bone fused {fused:?} not in {bounds:?}"
    );
}

#[test]
fn large_intermediates_are_contained_or_explicitly_unsafe() {
    let x = f32::from_bits(1.0f32.to_bits() + 1);
    let mesh = mesh_with(&[[x, 0.0, 0.0]], &[[0, 0, 0, 0]], &[[1.0, 0.0, 0.0, 0.0]]);
    let envelope = SkinEnvelope::from_mesh(&mesh).unwrap();
    let large = Mat4::from_cols(
        Vec4::new(1.0e30, 0.0, 0.0, 0.0),
        Vec4::new(0.0, 1.0, 0.0, 0.0),
        Vec4::new(0.0, 0.0, 1.0, 0.0),
        Vec4::new(-1.0e30, 0.0, 0.0, 1.0),
    );
    match envelope.posed_bounds(|_| Some(large)) {
        Ok(bounds) => {
            let palette = [large, Mat4::IDENTITY, Mat4::IDENTITY, Mat4::IDENTITY];
            let weights = [1.0, 0.0, 0.0, 0.0];
            let separate = shader_order_skin_separate_f32(weights, palette, Vec3::new(x, 0.0, 0.0));
            let fused = shader_order_skin_fused_f32(weights, palette, Vec3::new(x, 0.0, 0.0));
            assert!(bounds.contains(separate, 0.0), "large {bounds:?}");
            assert!(bounds.contains(fused, 0.0), "large {bounds:?}");
        }
        Err(error) => assert_eq!(error, SkinBoundsError::NonFiniteBounds),
    }

    let overflowing = Mat4::from_cols(
        Vec4::new(f32::MAX, 0.0, 0.0, 0.0),
        Vec4::new(0.0, 1.0, 0.0, 0.0),
        Vec4::new(0.0, 0.0, 1.0, 0.0),
        Vec4::new(-f32::MAX, 0.0, 0.0, 1.0),
    );
    assert_eq!(
        envelope.posed_bounds(|_| Some(overflowing)).err(),
        Some(SkinBoundsError::NonFiniteBounds)
    );
}

#[test]
fn f32_endpoint_conversion_moves_outward_for_both_signs() {
    for value in [1.0f64, -1.0, 3.5, -3.5, 1.0e30, -1.0e30] {
        let down = outward_down(value).unwrap();
        let up = outward_up(value).unwrap();
        assert!(down < value as f32, "down {down} for {value}");
        assert!(up > value as f32, "up {up} for {value}");
    }
}

#[test]
fn known_static_mesh_is_not_skinned_and_builds_no_envelope() {
    let mut mesh = Mesh::new(
        PrimitiveTopology::TriangleList,
        RenderAssetUsages::default(),
    );
    mesh.insert_attribute(
        Mesh::ATTRIBUTE_POSITION,
        VertexAttributeValues::Float32x3(vec![[0.0, 0.0, 0.0]]),
    );
    let mut cache = SkinEnvelopeCache::default();
    let palettes = palette();
    assert_eq!(
        skinned_result(id(2), &mesh, 1, Some(&palettes), &mut cache),
        SkinBoundsResult::NotSkinned
    );
    assert_eq!(cache.builds(), 0);
    assert!(!cache.contains(id(2)));
}

#[test]
fn actual_skin_attributes_select_the_skinned_path() {
    let mesh = mesh_with(
        &[[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        &[[0, 1, 0, 0], [1, 1, 1, 1], [0, 0, 0, 0]],
        &[[0.5, 0.5, 0.0, 0.0], [0.25; 4], [0.0; 4]],
    );
    let mut cache = SkinEnvelopeCache::default();
    let palettes = palette();
    let result = skinned_result(id(1), &mesh, 1, Some(&palettes), &mut cache);
    let SkinBoundsResult::Skinned {
        rig_origin,
        relative_bounds,
    } = result
    else {
        panic!("expected skinned result: {result:?}");
    };
    assert_eq!(rig_origin, Vec3::new(9000.0, 10.0, -4.0));
    assert!(relative_bounds.is_finite());
    assert!(relative_bounds.contains(Vec3::ZERO, 1.0e-3));
    assert_eq!(cache.builds(), 1);
    assert!(cache.contains(id(1)));
}

#[test]
fn direct_weighted_vertices_remain_inside_the_posed_envelope() {
    let positions = vec![
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.5, 0.5, 0.5],
    ];
    let joints = vec![[0, 1, 0, 0], [1, 1, 1, 1], [0, 0, 0, 0], [1, 0, 1, 0]];
    let weights = vec![
        [0.5, 0.25, 0.0, 0.0],
        [0.25; 4],
        [0.0; 4],
        [0.75, 0.75, 0.0, 0.0],
    ];
    let mesh = mesh_with(&positions, &joints, &weights);
    let envelope = SkinEnvelope::from_mesh(&mesh).expect("valid envelope");
    let m0 = Mat4::from_scale_rotation_translation(
        Vec3::new(2.0, 0.5, 1.5),
        Quat::from_rotation_x(0.7),
        Vec3::new(3.0, -2.0, 1.0),
    );
    let m1 =
        Mat4::from_rotation_translation(Quat::from_rotation_z(-0.4), Vec3::new(-1.0, 4.0, 2.0));
    let bounds = envelope
        .posed_bounds(|bone| match bone {
            0 => Some(m0),
            1 => Some(m1),
            _ => None,
        })
        .expect("posed bounds");

    for i in 0..positions.len() {
        let position = Vec3::from_array(positions[i]);
        let mut direct = Vec3::ZERO;
        for k in 0..4 {
            let matrix = match joints[i][k] {
                0 => m0,
                1 => m1,
                _ => unreachable!(),
            };
            direct += matrix.transform_point3(position) * weights[i][k];
        }
        assert!(
            bounds.contains(direct, 1.0e-3),
            "vertex {i} direct {direct:?} not in {bounds:?}"
        );
        let influence = std::array::from_fn(|k| match joints[i][k] {
            0 => m0,
            1 => m1,
            _ => unreachable!(),
        });
        let separate = shader_order_skin_separate_f32(weights[i], influence, position);
        let fused = shader_order_skin_fused_f32(weights[i], influence, position);
        assert!(
            bounds.contains(separate, 0.0),
            "vertex {i} separate {separate:?} not in {bounds:?}"
        );
        assert!(
            bounds.contains(fused, 0.0),
            "vertex {i} fused {fused:?} not in {bounds:?}"
        );
    }
}

#[test]
fn malformed_and_nonfinite_meshes_are_unsafe() {
    let mut missing = Mesh::new(
        PrimitiveTopology::TriangleList,
        RenderAssetUsages::default(),
    );
    missing.insert_attribute(
        Mesh::ATTRIBUTE_POSITION,
        VertexAttributeValues::Float32x3(vec![[0.0; 3]]),
    );
    assert_eq!(
        SkinEnvelope::from_mesh(&missing).err(),
        Some(SkinBoundsError::MissingJointIndices)
    );

    let nonfinite = mesh_with(
        &[[f32::NAN, 0.0, 0.0]],
        &[[0, 0, 0, 0]],
        &[[1.0, 0.0, 0.0, 0.0]],
    );
    assert_eq!(
        SkinEnvelope::from_mesh(&nonfinite).err(),
        Some(SkinBoundsError::NonFinitePosition)
    );

    let bad_weights = mesh_with(&[[0.0; 3]], &[[0, 0, 0, 0]], &[[-1.0, 0.0, 0.0, 0.0]]);
    assert_eq!(
        SkinEnvelope::from_mesh(&bad_weights).err(),
        Some(SkinBoundsError::NegativeWeight)
    );

    let mismatch = mesh_with(
        &[[0.0; 3], [1.0; 3]],
        &[[0, 0, 0, 0]],
        &[[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]],
    );
    assert_eq!(
        SkinEnvelope::from_mesh(&mismatch).err(),
        Some(SkinBoundsError::AttributeLengthMismatch)
    );
}

#[test]
fn pose_rejects_invalid_palette_rows() {
    let mesh = mesh_with(&[[0.0; 3]], &[[1, 0, 0, 0]], &[[1.0, 0.0, 0.0, 0.0]]);
    let envelope = SkinEnvelope::from_mesh(&mesh).unwrap();
    assert_eq!(
        envelope.posed_bounds(|_| None).err(),
        Some(SkinBoundsError::InvalidPalette)
    );
    let not_affine = Mat4::from_cols(Vec4::X, Vec4::Y, Vec4::Z, Vec4::new(0.0, 0.0, 0.0, 2.0));
    assert_eq!(
        envelope.posed_bounds(|_| Some(not_affine)).err(),
        Some(SkinBoundsError::InvalidPalette)
    );
}

#[test]
fn invalid_envelopes_are_cached_until_the_mesh_changes() {
    let mut mesh = Mesh::new(
        PrimitiveTopology::TriangleList,
        RenderAssetUsages::default(),
    );
    mesh.insert_attribute(
        Mesh::ATTRIBUTE_POSITION,
        VertexAttributeValues::Float32x3(vec![[0.0; 3]]),
    );
    let mut cache = SkinEnvelopeCache::default();
    let id = id(11);
    assert_eq!(
        cache.envelope(id, &mesh).err(),
        Some(SkinBoundsError::MissingJointIndices)
    );
    assert_eq!(cache.builds(), 1);
    assert_eq!(
        cache.envelope(id, &mesh).err(),
        Some(SkinBoundsError::MissingJointIndices)
    );
    assert_eq!(cache.builds(), 1, "invalid result stays cached");
    cache.evict(id);
    assert_eq!(
        cache.envelope(id, &mesh).err(),
        Some(SkinBoundsError::MissingJointIndices)
    );
    assert_eq!(cache.builds(), 2, "asset change rebuilds even on error");
}

#[test]
fn mesh_modification_rebuilds_and_retirement_evicts_the_cache() {
    let mesh = mesh_with(&[[0.0; 3]], &[[0, 0, 0, 0]], &[[1.0, 0.0, 0.0, 0.0]]);
    let mut cache = SkinEnvelopeCache::default();
    let id = id(5);
    assert!(cache.envelope(id, &mesh).is_ok());
    assert_eq!(cache.builds(), 1);
    assert!(cache.envelope(id, &mesh).is_ok());
    assert_eq!(cache.builds(), 1, "unchanged mesh reuses the envelope");
    cache.evict(id);
    assert!(!cache.contains(id));
    assert!(cache.envelope(id, &mesh).is_ok());
    assert_eq!(cache.builds(), 2, "modified/removed mesh rebuilds");
    cache.retain_referenced(&HashSet::new());
    assert!(!cache.contains(id));
}

#[test]
fn repeated_poses_reuse_the_mesh_envelope_but_recompute_bounds() {
    let mesh = mesh_with(&[[0.0, 0.0, 0.0]], &[[0, 0, 0, 0]], &[[1.0, 0.0, 0.0, 0.0]]);
    let mut cache = SkinEnvelopeCache::default();
    let id = id(6);
    let _ = cache.envelope(id, &mesh).unwrap();
    assert_eq!(cache.builds(), 1);
    let palettes = palette();
    let first = skinned_result(id, &mesh, 1, Some(&palettes), &mut cache);
    let second = skinned_result(id, &mesh, 1, Some(&palettes), &mut cache);
    assert_eq!(cache.builds(), 1);
    assert_eq!(first, second);
}

#[test]
fn missing_freed_and_reused_rig_slots_do_not_preserve_old_bounds() {
    let mesh = mesh_with(&[[0.0, 0.0, 0.0]], &[[0, 0, 0, 0]], &[[1.0, 0.0, 0.0, 0.0]]);
    let mut cache = SkinEnvelopeCache::default();
    let mut palettes = palette();
    let slot = 1u16;
    let old = skinned_result(id(7), &mesh, slot, Some(&palettes), &mut cache);
    assert!(matches!(old, SkinBoundsResult::Skinned { .. }));
    palettes.free(slot);
    let freed = skinned_result(id(7), &mesh, slot, Some(&palettes), &mut cache);
    assert_eq!(
        freed,
        SkinBoundsResult::Unsafe(SkinBoundsError::InvalidPalette)
    );
    let (reused, _) = palettes.alloc(2).unwrap();
    assert_eq!(reused, slot);
    palettes.write_rider(
        reused,
        Affine3A::from_translation(Vec3::new(-5.0, 1.0, 2.0)),
        Vec3::new(100.0, 0.0, 0.0),
    );
    let new = skinned_result(id(7), &mesh, reused, Some(&palettes), &mut cache);
    assert!(matches!(new, SkinBoundsResult::Skinned { .. }));
    assert_ne!(old, new);
}

#[test]
fn large_origin_stays_separate_from_small_relative_bounds() {
    let mesh = mesh_with(&[[0.0, 0.0, 0.0]], &[[0, 0, 0, 0]], &[[1.0, 0.0, 0.0, 0.0]]);
    let mut cache = SkinEnvelopeCache::default();
    let palettes = palette();
    let result = skinned_result(id(8), &mesh, 1, Some(&palettes), &mut cache);
    let SkinBoundsResult::Skinned {
        rig_origin,
        relative_bounds,
    } = result
    else {
        panic!("expected skinned");
    };
    assert_eq!(rig_origin, Vec3::new(9000.0, 10.0, -4.0));
    assert!(relative_bounds.max.length() < 10.0);
}

#[test]
fn temporal_model_entity_publishes_skinned_component() {
    let mut app = App::new();
    app.insert_resource(crate::view::MsaaSetting { samples: 1 })
        .init_resource::<crate::view::Viewer>()
        .init_resource::<crate::schedule::WorldLive>()
        .add_message::<crate::world_map::MapChange>()
        .add_plugins(super::super::AaPlugin)
        .insert_resource(Assets::<Mesh>::default());
    let mut palettes = RigPalettes::default();
    let (slot, _) = palettes.alloc(2).unwrap();
    palettes.write_rider(
        slot,
        Affine3A::from_translation(Vec3::new(1.0, 2.0, 3.0)),
        Vec3::new(9000.0, 10.0, -4.0),
    );
    app.insert_resource(palettes);
    let mesh = mesh_with(
        &[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
        &[[0, 1, 0, 0], [1, 0, 0, 0]],
        &[[0.5, 0.5, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]],
    );
    let handle = app.world_mut().resource_mut::<Assets<Mesh>>().add(mesh);
    app.world_mut().spawn((
        WorldCamera,
        Camera {
            is_active: true,
            ..default()
        },
        Msaa::Off,
    ));
    let entity = app
        .world_mut()
        .spawn((
            Mesh3d(handle.clone()),
            MeshMaterial3d(Handle::<WowModelMaterial>::default()),
            MeshTag(crate::mesh_tag::spawn_tag(slot, 1.0)),
        ))
        .id();
    let hidden = app
        .world_mut()
        .spawn((
            Mesh3d(handle.clone()),
            MeshMaterial3d(Handle::<WowModelMaterial>::default()),
            MeshTag(crate::mesh_tag::spawn_tag(slot, 1.0)),
            ViewVisibility::HIDDEN,
            TemporalSkinBounds {
                mesh: handle.id(),
                slot,
                result: SkinBoundsResult::NotSkinned,
            },
        ))
        .id();
    let untagged = app
        .world_mut()
        .spawn((
            Mesh3d(handle.clone()),
            MeshMaterial3d(Handle::<WowModelMaterial>::default()),
        ))
        .id();
    app.world_mut()
        .resource_mut::<AaPolicy>()
        .request(AaMode::Taa);
    app.world_mut().resource_mut::<AaReadiness>().taa = true;
    app.update();

    let bounds = app
        .world()
        .get::<TemporalSkinBounds>(entity)
        .expect("temporal model metadata");
    assert!(matches!(
        bounds.result,
        SkinBoundsResult::Skinned {
            rig_origin,
            relative_bounds,
        } if rig_origin == Vec3::new(9000.0, 10.0, -4.0)
            && relative_bounds.is_finite()
    ));
    assert!(app.world().get::<TemporalSkinBounds>(hidden).is_none());
    assert_eq!(
        app.world()
            .get::<TemporalSkinBounds>(untagged)
            .map(|bounds| &bounds.result),
        Some(&SkinBoundsResult::Unsafe(SkinBoundsError::MissingTag))
    );
    assert_eq!(app.world().resource::<SkinEnvelopeCache>().builds(), 1);

    let replacement = mesh_with(&[[2.0, 0.0, 0.0]], &[[1, 0, 0, 0]], &[[1.0, 0.0, 0.0, 0.0]]);
    let replacement = app
        .world_mut()
        .resource_mut::<Assets<Mesh>>()
        .add(replacement);
    app.world_mut()
        .entity_mut(entity)
        .get_mut::<Mesh3d>()
        .unwrap()
        .0 = replacement.clone();
    app.update();
    let bounds = app
        .world()
        .get::<TemporalSkinBounds>(entity)
        .expect("replacement metadata");
    assert_eq!(bounds.mesh, replacement.id());
    assert_eq!(app.world().resource::<SkinEnvelopeCache>().builds(), 2);
}

#[test]
fn missing_slot_palette_and_origin_are_unsafe() {
    let mesh = mesh_with(&[[0.0; 3]], &[[0, 0, 0, 0]], &[[1.0, 0.0, 0.0, 0.0]]);
    let mut cache = SkinEnvelopeCache::default();
    let palettes = palette();
    assert_eq!(
        skinned_result(id(10), &mesh, 0, Some(&palettes), &mut cache),
        SkinBoundsResult::Unsafe(SkinBoundsError::MissingRigSlot)
    );
    assert_eq!(
        skinned_result(id(10), &mesh, 1, None, &mut cache),
        SkinBoundsResult::Unsafe(SkinBoundsError::MissingPalette)
    );
    assert_eq!(
        skinned_result(id(10), &mesh, u16::MAX, Some(&palettes), &mut cache),
        SkinBoundsResult::Unsafe(SkinBoundsError::MissingOrigin)
    );

    let mut bad_origin = RigPalettes::default();
    let (slot, _) = bad_origin.alloc(1).unwrap();
    bad_origin.write_rider(slot, Affine3A::IDENTITY, Vec3::new(f32::NAN, 0.0, 0.0));
    assert_eq!(
        skinned_result(id(10), &mesh, slot, Some(&bad_origin), &mut cache),
        SkinBoundsResult::Unsafe(SkinBoundsError::NonFiniteRigOrigin)
    );
}

#[test]
fn dormant_policy_removes_metadata_and_does_not_build_envelopes() {
    let mut app = App::new();
    app.insert_resource(crate::view::MsaaSetting { samples: 1 })
        .init_resource::<crate::view::Viewer>()
        .init_resource::<crate::schedule::WorldLive>()
        .add_message::<crate::world_map::MapChange>()
        .add_plugins(super::super::AaPlugin);
    let entity = app
        .world_mut()
        .spawn(TemporalSkinBounds {
            mesh: id(9),
            slot: 0,
            result: SkinBoundsResult::NotSkinned,
        })
        .id();
    app.update();
    assert!(app.world().get::<TemporalSkinBounds>(entity).is_none());
    assert_eq!(
        app.world().resource::<SkinEnvelopeCache>().builds(),
        0,
        "dormant path builds nothing"
    );
}

#[test]
fn headless_plugin_does_not_require_mesh_assets() {
    let mut app = App::new();
    app.insert_resource(crate::view::MsaaSetting { samples: 1 })
        .init_resource::<crate::view::Viewer>()
        .init_resource::<crate::schedule::WorldLive>()
        .add_message::<crate::world_map::MapChange>()
        .add_plugins(super::super::AaPlugin);
    app.update();
}
