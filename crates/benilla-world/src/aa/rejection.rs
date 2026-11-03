//! Conservative per-view rejection masks for entity models and selected retained wind foliage.
//!
//! The mask is `1` where an entity model may appear, or where a selected retained static-gx wind
//! region may be displaced, and must not reuse camera-only temporal history. `0` means only that
//! this lane found no rejection, not that the pixel is temporally safe. The lane deliberately uses
//! projected rectangles rather than replaying materials or trusting camera-only object motion.
//!
//! Unknown, unbounded, wind-displaced or inconsistent visible model data produces a full-view
//! rejection. Known `SKY_DEPTH_MARKER` content is the narrow exception: it pins background depth 0,
//! which camera-motion reconstruction already rejects, so it is excluded from this model-mask lane.
//! That exclusion does not generalize to any other material. Retained wind coverage is coarse and
//! may include nearby rigid pixels; foliage velocity, per-leaf tightness, animated materials and
//! later atmospheric coverage remain outside this lane. The lane retains one current/previous mask
//! pair per view; it does not create color/depth reconstruction history.

use std::any::TypeId;
use std::collections::HashSet;

use benilla_assets::materials::WowModelMaterial;
use bevy::asset::Assets;
use bevy::camera::primitives::Aabb;
use bevy::camera::visibility::{RenderLayers, VisibleEntities};
use bevy::core_pipeline::core_3d::graph::{Core3d, Node3d};
use bevy::ecs::query::QueryItem;
use bevy::image::ToExtents;
use bevy::math::{DVec2, DVec3, DVec4, URect};
use bevy::mesh::{Mesh3d, MeshTag};
use bevy::platform::collections::HashMap;
use bevy::prelude::*;
use bevy::render::extract_resource::extract_resource;
use bevy::render::render_graph::{
    NodeRunError, RenderGraphContext, RenderGraphExt, RenderLabel, ViewNode, ViewNodeRunner,
};
use bevy::render::render_resource::*;
use bevy::render::renderer::{RenderContext, RenderDevice};
use bevy::render::sync_world::RenderEntity;
use bevy::render::texture::CachedTexture;
use bevy::render::{Extract, ExtractSchedule, Render, RenderApp, RenderStartup, RenderSystems};

use super::depth::{self, DepthKey, TemporalDepthRequests};
use super::frame_pair::TemporalFramePair;
use super::reactive::TemporalOpaqueSnapshotLabel;
use super::skin_bounds::{Bounds3, SkinBoundsResult, TemporalSkinBounds};
use crate::lighting::MonkeyFrame;
use crate::mesh_tag::rig_of;
use crate::model_render::{FOLIAGE_WIND_MARKER, SKY_DEPTH_MARKER};
use crate::static_gx::{RetainedTemporalSource, TemporalWindRegion};
use crate::view::WorldCamera;

const MAX_RECTS_PER_VIEW: usize = 256;
const RECT_PADDING_PX: f64 = 2.0;
const NEAR_W_EPSILON: f64 = 1.0e-6;

#[derive(Clone, Debug, PartialEq, Eq)]
enum ProjectedReject {
    Outside,
    Rect(URect),
    FullView,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct ViewModelRejects {
    key: DepthKey,
    full_view: bool,
    rects: Vec<URect>,
}

impl ViewModelRejects {
    fn new(key: DepthKey) -> Self {
        Self {
            key,
            full_view: false,
            rects: Vec::new(),
        }
    }

    fn reset(&mut self, key: DepthKey) {
        self.key = key;
        self.full_view = false;
        self.rects.clear();
    }

    fn reject_full(&mut self) {
        self.full_view = true;
        self.rects.clear();
    }
}

#[derive(Resource, Default)]
struct ModelRejectRequests {
    epoch: u64,
    by_render: HashMap<Entity, ViewModelRejects>,
}

fn outward_down(value: f64) -> Option<f32> {
    let value = value as f32;
    value.is_finite().then(|| value.next_down())
}

fn outward_up(value: f64) -> Option<f32> {
    let value = value as f32;
    value.is_finite().then(|| value.next_up())
}

fn finite_bounds(bounds: &Bounds3) -> bool {
    bounds.min.is_finite()
        && bounds.max.is_finite()
        && bounds.min.x <= bounds.max.x
        && bounds.min.y <= bounds.max.y
        && bounds.min.z <= bounds.max.z
}

fn clamp_pixel(value: f64, extent: u32) -> u32 {
    if value <= 0.0 {
        0
    } else if value >= extent as f64 {
        extent
    } else {
        value as u32
    }
}

/// Project a rig-relative/world-origin bound through the current camera. Invalid or ambiguous
/// input is deliberately `FullView`; the only `Outside` results are bounds wholly outside one
/// homogeneous frustum plane.
fn project_reject_bounds(
    origin: Vec3,
    bounds: Bounds3,
    camera_world: Mat4,
    clip_from_view: Mat4,
    extent: UVec2,
) -> ProjectedReject {
    if !origin.is_finite()
        || !finite_bounds(&bounds)
        || extent.x == 0
        || extent.y == 0
        || !camera_world.is_finite()
        || !clip_from_view.is_finite()
        || camera_world.determinant().abs() < 1.0e-12
        || clip_from_view.determinant().abs() < 1.0e-12
    {
        return ProjectedReject::FullView;
    }
    let camera_view_from_world = camera_world.as_dmat4().inverse();
    if !camera_view_from_world
        .to_cols_array()
        .iter()
        .all(|value| value.is_finite())
    {
        return ProjectedReject::FullView;
    }
    let origin = DVec3::new(origin.x as f64, origin.y as f64, origin.z as f64);
    let min = bounds.min;
    let max = bounds.max;
    let xs = [min.x as f64, max.x as f64];
    let ys = [min.y as f64, max.y as f64];
    let zs = [min.z as f64, max.z as f64];
    let clip_from_view = clip_from_view.as_dmat4();
    let mut clips = [DVec4::ZERO; 8];
    let mut index = 0usize;
    let mut all_outside = [true; 6];
    for x in xs {
        for y in ys {
            for z in zs {
                let world = origin + DVec3::new(x, y, z);
                let view = camera_view_from_world.transform_point3(world);
                let clip = clip_from_view * DVec4::new(view.x, view.y, view.z, 1.0);
                if !clip.is_finite() {
                    return ProjectedReject::FullView;
                }
                all_outside[0] &= clip.x > clip.w;
                all_outside[1] &= clip.x < -clip.w;
                all_outside[2] &= clip.y > clip.w;
                all_outside[3] &= clip.y < -clip.w;
                all_outside[4] &= clip.z < 0.0;
                all_outside[5] &= clip.z > clip.w;
                clips[index] = clip;
                index += 1;
            }
        }
    }
    if all_outside.iter().any(|outside| *outside) {
        return ProjectedReject::Outside;
    }
    for clip in clips {
        if clip.w <= NEAR_W_EPSILON || clip.z < 0.0 || clip.z > clip.w {
            return ProjectedReject::FullView;
        }
    }

    let mut uv_min = DVec2::splat(f64::INFINITY);
    let mut uv_max = DVec2::splat(f64::NEG_INFINITY);
    for clip in clips {
        let ndc_x = clip.x / clip.w;
        let ndc_y = clip.y / clip.w;
        if !ndc_x.is_finite() || !ndc_y.is_finite() {
            return ProjectedReject::FullView;
        }
        let uv = DVec2::new(0.5 * ndc_x + 0.5, 0.5 - 0.5 * ndc_y);
        uv_min = uv_min.min(uv);
        uv_max = uv_max.max(uv);
    }
    let extent_x = extent.x as f64;
    let extent_y = extent.y as f64;
    let min_x = (uv_min.x * extent_x - RECT_PADDING_PX).floor();
    let min_y = (uv_min.y * extent_y - RECT_PADDING_PX).floor();
    let max_x = (uv_max.x * extent_x + RECT_PADDING_PX).ceil();
    let max_y = (uv_max.y * extent_y + RECT_PADDING_PX).ceil();
    if !min_x.is_finite() || !min_y.is_finite() || !max_x.is_finite() || !max_y.is_finite() {
        return ProjectedReject::FullView;
    }
    let min = UVec2::new(clamp_pixel(min_x, extent.x), clamp_pixel(min_y, extent.y));
    let max = UVec2::new(clamp_pixel(max_x, extent.x), clamp_pixel(max_y, extent.y));
    if min.x >= max.x || min.y >= max.y {
        return ProjectedReject::Outside;
    }
    ProjectedReject::Rect(URect { min, max })
}

/// Expand a retained wind region by the inspected tree-offset envelope, then project it through
/// the same conservative bound path as entity models. The envelope covers the current wind hook's
/// directional wave sum, crown weight, roll bound, gain and vertical quadratic; it is not a claim
/// about arbitrary future shaders or malformed vertices.
fn project_retained_wind(
    region: TemporalWindRegion,
    frame: Option<&MonkeyFrame>,
    camera_world: Mat4,
    clip_from_view: Mat4,
    extent: UVec2,
) -> ProjectedReject {
    let origin = region.origin;
    let center = region.bounds.center;
    let half = region.bounds.half_extents;
    if !origin.is_finite() || !center.is_finite() || !half.is_finite() || half.min_element() < 0.0 {
        return ProjectedReject::FullView;
    }
    let Some(frame) = frame else {
        return ProjectedReject::FullView;
    };
    let [wind_x, wind_z] = frame.wind_dir;
    let direction = DVec2::new(wind_x as f64 + 1.0e-6, wind_z as f64);
    if !direction.is_finite()
        || direction.length_squared() <= 1.0e-12
        || !frame.wind_base_heading.is_finite()
        || !frame.wind_travel.is_finite()
        || !frame.wind_gust.is_finite()
        || !frame.sway_strength.is_finite()
        || !frame.tree_strength.is_finite()
    {
        return ProjectedReject::FullView;
    }
    let horizontal = 0.39
        * 1.3
        * (0.75 + 0.5 * frame.wind_gust as f64).abs()
        * 1.05
        * frame.sway_strength.abs() as f64
        * frame.tree_strength.abs() as f64;
    let vertical = 0.15 * horizontal * horizontal;
    if !horizontal.is_finite() || !vertical.is_finite() {
        return ProjectedReject::FullView;
    }

    let origin64 = DVec3::new(origin.x as f64, origin.y as f64, origin.z as f64);
    let center64 = DVec3::new(center.x as f64, center.y as f64, center.z as f64);
    let half64 = DVec3::new(half.x as f64, half.y as f64, half.z as f64);
    let displacement = DVec3::new(horizontal, vertical, horizontal);
    let guard = DVec3::new(
        128.0
            * f32::EPSILON as f64
            * (origin64.x.abs() + center64.x.abs() + half64.x + displacement.x + 1.0),
        128.0
            * f32::EPSILON as f64
            * (origin64.y.abs() + center64.y.abs() + half64.y + displacement.y + 1.0),
        128.0
            * f32::EPSILON as f64
            * (origin64.z.abs() + center64.z.abs() + half64.z + displacement.z + 1.0),
    );
    let min = center64 - half64 - displacement - guard;
    let max = center64 + half64 + displacement + guard;
    if !min.is_finite() || !max.is_finite() || min.x > max.x || min.y > max.y || min.z > max.z {
        return ProjectedReject::FullView;
    }
    let (Some(min_x), Some(min_y), Some(min_z)) = (
        outward_down(min.x),
        outward_down(min.y),
        outward_down(min.z),
    ) else {
        return ProjectedReject::FullView;
    };
    let (Some(max_x), Some(max_y), Some(max_z)) =
        (outward_up(max.x), outward_up(max.y), outward_up(max.z))
    else {
        return ProjectedReject::FullView;
    };
    let bounds = Bounds3 {
        min: Vec3::new(min_x, min_y, min_z),
        max: Vec3::new(max_x, max_y, max_z),
    };
    project_reject_bounds(origin, bounds, camera_world, clip_from_view, extent)
}

fn project_rigid_aabb(
    aabb: &Aabb,
    transform: &GlobalTransform,
    camera_world: Mat4,
    clip_from_view: Mat4,
    extent: UVec2,
) -> ProjectedReject {
    let center = Vec3::from(aabb.center);
    let half = Vec3::from(aabb.half_extents);
    if !center.is_finite()
        || !half.is_finite()
        || half.min_element() < 0.0
        || !transform.to_matrix().is_finite()
    {
        return ProjectedReject::FullView;
    }
    let origin = transform.translation();
    if !origin.is_finite() {
        return ProjectedReject::FullView;
    }
    let matrix = transform.to_matrix().as_dmat4();
    let origin64 = DVec3::new(origin.x as f64, origin.y as f64, origin.z as f64);
    let mut min = DVec3::splat(f64::INFINITY);
    let mut max = DVec3::splat(f64::NEG_INFINITY);
    for sx in [-1.0f64, 1.0] {
        for sy in [-1.0f64, 1.0] {
            for sz in [-1.0f64, 1.0] {
                let local = DVec3::new(
                    center.x as f64 + sx * half.x as f64,
                    center.y as f64 + sy * half.y as f64,
                    center.z as f64 + sz * half.z as f64,
                );
                let relative = matrix.transform_point3(local) - origin64;
                if !relative.is_finite() {
                    return ProjectedReject::FullView;
                }
                min = min.min(relative);
                max = max.max(relative);
            }
        }
    }
    let Some(min_x) = outward_down(min.x) else {
        return ProjectedReject::FullView;
    };
    let Some(min_y) = outward_down(min.y) else {
        return ProjectedReject::FullView;
    };
    let Some(min_z) = outward_down(min.z) else {
        return ProjectedReject::FullView;
    };
    let Some(max_x) = outward_up(max.x) else {
        return ProjectedReject::FullView;
    };
    let Some(max_y) = outward_up(max.y) else {
        return ProjectedReject::FullView;
    };
    let Some(max_z) = outward_up(max.z) else {
        return ProjectedReject::FullView;
    };
    let bounds = Bounds3 {
        min: Vec3::new(min_x, min_y, min_z),
        max: Vec3::new(max_x, max_y, max_z),
    };
    project_reject_bounds(origin, bounds, camera_world, clip_from_view, extent)
}

fn push_rect(view: &mut ViewModelRejects, rect: URect) -> bool {
    if view.rects.len() >= MAX_RECTS_PER_VIEW {
        view.reject_full();
        return false;
    }
    view.rects.push(rect);
    true
}

fn layer_visible(camera: Option<&RenderLayers>, model: Option<&RenderLayers>) -> bool {
    let camera = camera.cloned().unwrap_or_default();
    let model = model.cloned().unwrap_or_default();
    camera.intersects(&model)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum MaterialDisposition {
    Clear,
    SkyOnly,
    FullView,
}

fn material_disposition(material: &WowModelMaterial) -> MaterialDisposition {
    marker_disposition(material.extension.clutter_fade)
}

fn marker_disposition(fade: Vec4) -> MaterialDisposition {
    if !fade.is_finite() || fade.z < 0.0 || fade.z > u16::MAX as f32 || fade.z.fract() != 0.0 {
        return MaterialDisposition::FullView;
    }
    let marker = fade.z as u32;
    if fade.w > 0.5 || marker & u32::from(FOLIAGE_WIND_MARKER) != 0 {
        return MaterialDisposition::FullView;
    }
    if marker & u32::from(SKY_DEPTH_MARKER) != 0 {
        return MaterialDisposition::SkyOnly;
    }
    MaterialDisposition::Clear
}

fn push_projected(view: &mut ViewModelRejects, projected: ProjectedReject) -> bool {
    match projected {
        ProjectedReject::Outside => true,
        ProjectedReject::Rect(rect) => push_rect(view, rect),
        ProjectedReject::FullView => {
            view.reject_full();
            false
        }
    }
}

fn metadata_consistent(bounds: &TemporalSkinBounds, mesh: AssetId<Mesh>, slot: u16) -> bool {
    bounds.mesh == mesh && bounds.slot == slot
}

fn project_model_metadata(
    bounds: &TemporalSkinBounds,
    aabb: Option<&Aabb>,
    transform: Option<&GlobalTransform>,
    camera_world: Mat4,
    clip_from_view: Mat4,
    extent: UVec2,
) -> ProjectedReject {
    match &bounds.result {
        SkinBoundsResult::Unsafe(_) => ProjectedReject::FullView,
        SkinBoundsResult::Skinned {
            rig_origin,
            relative_bounds,
        } => project_reject_bounds(
            *rig_origin,
            *relative_bounds,
            camera_world,
            clip_from_view,
            extent,
        ),
        SkinBoundsResult::NotSkinned => {
            let (Some(aabb), Some(transform)) = (aabb, transform) else {
                return ProjectedReject::FullView;
            };
            project_rigid_aabb(aabb, transform, camera_world, clip_from_view, extent)
        }
    }
}

/// Extract one rejection plan per depth request. This runs after the depth extraction so the epoch
/// and render-entity map are current; it never introduces another clock.
#[allow(clippy::type_complexity)] // Bevy query tuples read better inline.
fn extract_model_rejections(
    mut requests: ResMut<ModelRejectRequests>,
    depth: Res<TemporalDepthRequests>,
    retained_main: Extract<Option<Res<RetainedTemporalSource>>>,
    retained_render: Option<Res<RetainedTemporalSource>>,
    monkey_frame: Extract<Option<Res<MonkeyFrame>>>,
    materials: Extract<Res<Assets<WowModelMaterial>>>,
    cameras: Extract<
        Query<
            (
                &RenderEntity,
                &Camera,
                Option<&VisibleEntities>,
                Option<&RenderLayers>,
                &GlobalTransform,
            ),
            With<WorldCamera>,
        >,
    >,
    models: Extract<
        Query<(
            Entity,
            &Mesh3d,
            Option<&MeshTag>,
            Option<&MeshMaterial3d<WowModelMaterial>>,
            Option<&Aabb>,
            Option<&GlobalTransform>,
            Option<&TemporalSkinBounds>,
            Option<&RenderLayers>,
        )>,
    >,
) {
    requests.epoch = depth.epoch();
    let mut old = std::mem::take(&mut requests.by_render);
    let mut seen = HashSet::new();
    for (render_entity, camera, visible, camera_layers, camera_world) in cameras.iter() {
        let render = render_entity.id();
        let Some(depth_request) = depth.get(render) else {
            continue;
        };
        seen.insert(render);
        let mut view = old
            .remove(&render)
            .unwrap_or_else(|| ViewModelRejects::new(depth_request.key()));
        view.reset(depth_request.key());
        if !camera.is_active {
            view.reject_full();
            requests.by_render.insert(render, view);
            continue;
        }
        let Some(visible) = visible else {
            view.reject_full();
            requests.by_render.insert(render, view);
            continue;
        };
        let camera_world = camera_world.to_matrix();
        let clip_from_view = camera.clip_from_view();
        if !camera_world.is_finite()
            || !clip_from_view.is_finite()
            || camera_world.determinant().abs() < 1.0e-12
            || clip_from_view.determinant().abs() < 1.0e-12
        {
            view.reject_full();
            requests.by_render.insert(render, view);
            continue;
        }
        let extent = depth_request.size;
        let retained = retained_main.as_deref().or(retained_render.as_deref());
        if let Some(source) = retained {
            let frame = monkey_frame.as_ref().map(|frame| &**frame);
            let mut retained_full = false;
            source.visit_temporal_wind(|region| {
                let Some(region) = region else {
                    retained_full = true;
                    return false;
                };
                let projected =
                    project_retained_wind(region, frame, camera_world, clip_from_view, extent);
                if !push_projected(&mut view, projected) {
                    retained_full = true;
                    return false;
                }
                true
            });
            if retained_full {
                view.reject_full();
                requests.by_render.insert(render, view);
                continue;
            }
        }
        for &entity in visible.get(TypeId::of::<Mesh3d>()) {
            let Ok((_, mesh3d, tag, material, aabb, transform, bounds, model_layers)) =
                models.get(entity)
            else {
                view.reject_full();
                break;
            };
            let Some(material) = material else {
                continue;
            };
            if !layer_visible(camera_layers, model_layers) {
                continue;
            }
            let Some(material_asset) = materials.get(material.0.id()) else {
                view.reject_full();
                break;
            };
            match material_disposition(material_asset) {
                MaterialDisposition::Clear => {}
                MaterialDisposition::SkyOnly => continue,
                MaterialDisposition::FullView => {
                    view.reject_full();
                    break;
                }
            }
            let Some(tag) = tag else {
                view.reject_full();
                break;
            };
            let Some(bounds) = bounds else {
                view.reject_full();
                break;
            };
            if !metadata_consistent(bounds, mesh3d.0.id(), rig_of(tag.0)) {
                view.reject_full();
                break;
            }
            let projected = project_model_metadata(
                bounds,
                aabb,
                transform,
                camera_world,
                clip_from_view,
                extent,
            );
            if !push_projected(&mut view, projected) {
                break;
            }
        }
        requests.by_render.insert(render, view);
    }

    for (render, depth_request) in depth.iter() {
        if seen.contains(&render) {
            continue;
        }
        let mut view = old
            .remove(&render)
            .unwrap_or_else(|| ViewModelRejects::new(depth_request.key()));
        view.reset(depth_request.key());
        view.reject_full();
        requests.by_render.insert(render, view);
    }
}

#[derive(Component)]
pub(super) struct ModelRejectOutput {
    key: DepthKey,
    masks: TemporalFramePair<CachedTexture>,
}

impl ModelRejectOutput {
    fn new(key: DepthKey, masks: [CachedTexture; 2]) -> Self {
        Self {
            key,
            masks: TemporalFramePair::new(key.generation, masks),
        }
    }

    pub(super) fn key(&self) -> DepthKey {
        self.key
    }

    pub(super) fn current_mask(&self, epoch: u64, generation: u64) -> Option<&TextureView> {
        self.masks
            .current(epoch, generation)
            .map(|mask| &mask.default_view)
    }

    /// The immediately preceding model-rejection mask, consumed by motion-candidate history.
    pub(super) fn previous_mask(&self, epoch: u64, generation: u64) -> Option<&TextureView> {
        self.masks
            .previous(epoch, generation)
            .map(|mask| &mask.default_view)
    }
}

fn mask_descriptor(size: UVec2) -> TextureDescriptor<'static> {
    TextureDescriptor {
        label: Some("temporal model rejection"),
        size: size.to_extents(),
        mip_level_count: 1,
        sample_count: 1,
        dimension: TextureDimension::D2,
        format: TextureFormat::R8Unorm,
        usage: TextureUsages::RENDER_ATTACHMENT | TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum RejectOutputDecision {
    Keep,
    Replace,
    Remove,
}

fn reject_output_decision(
    existing: Option<DepthKey>,
    requested: DepthKey,
    depth_key: Option<DepthKey>,
) -> RejectOutputDecision {
    if depth_key != Some(requested) {
        RejectOutputDecision::Remove
    } else if existing == Some(requested) {
        RejectOutputDecision::Keep
    } else {
        RejectOutputDecision::Replace
    }
}

#[allow(clippy::type_complexity)] // Bevy query tuples read better inline.
fn prepare_model_reject_outputs(
    mut commands: Commands,
    device: Res<RenderDevice>,
    requests: Res<ModelRejectRequests>,
    views: Query<(
        Entity,
        &depth::TemporalDepthTexture,
        Option<&ModelRejectOutput>,
    )>,
    outputs: Query<(Entity, &ModelRejectOutput)>,
) {
    for (render, request) in &requests.by_render {
        let Ok((entity, depth_texture, existing)) = views.get(*render) else {
            commands.entity(*render).remove::<ModelRejectOutput>();
            continue;
        };
        let decision = reject_output_decision(
            existing.map(|output| output.key),
            request.key,
            Some(depth_texture.key()),
        );
        match decision {
            RejectOutputDecision::Remove => {
                commands.entity(entity).remove::<ModelRejectOutput>();
            }
            RejectOutputDecision::Keep => {}
            RejectOutputDecision::Replace => {
                // 2 bytes/pixel per eligible view: two independent R8Unorm masks, excluding
                // allocator overhead.
                let mask_a = device.create_texture(&mask_descriptor(request.key.size));
                let mask_b = device.create_texture(&mask_descriptor(request.key.size));
                let mask_a = CachedTexture {
                    default_view: mask_a.create_view(&TextureViewDescriptor::default()),
                    texture: mask_a,
                };
                let mask_b = CachedTexture {
                    default_view: mask_b.create_view(&TextureViewDescriptor::default()),
                    texture: mask_b,
                };
                commands
                    .entity(entity)
                    .insert(ModelRejectOutput::new(request.key, [mask_a, mask_b]));
            }
        }
    }
    for (entity, _) in &outputs {
        if !requests.by_render.contains_key(&entity) {
            commands.entity(entity).remove::<ModelRejectOutput>();
        }
    }
}

#[derive(Resource)]
struct ModelRejectPipeline {
    pipeline: CachedRenderPipelineId,
}

fn init_model_reject_pipeline(
    mut commands: Commands,
    asset_server: Option<Res<AssetServer>>,
    cache: Res<PipelineCache>,
) {
    let Some(asset_server) = asset_server else {
        return;
    };
    let shader: Handle<Shader> =
        asset_server.load("embedded://benilla_world/shaders/temporal_rejection.wgsl");
    let pipeline = cache.queue_render_pipeline(RenderPipelineDescriptor {
        label: Some("temporal model rejection".into()),
        layout: vec![],
        vertex: VertexState {
            shader: shader.clone(),
            entry_point: Some("vertex".into()),
            ..default()
        },
        fragment: Some(FragmentState {
            shader,
            entry_point: Some("fragment".into()),
            targets: vec![Some(ColorTargetState {
                format: TextureFormat::R8Unorm,
                blend: None,
                write_mask: ColorWrites::ALL,
            })],
            ..default()
        }),
        primitive: PrimitiveState {
            topology: PrimitiveTopology::TriangleList,
            ..default()
        },
        multisample: MultisampleState {
            count: 1,
            ..default()
        },
        ..default()
    });
    commands.insert_resource(ModelRejectPipeline { pipeline });
}

#[derive(Debug, Hash, PartialEq, Eq, Clone, RenderLabel)]
struct ModelRejectLabel;

#[derive(Default)]
struct ModelRejectNode;

impl ViewNode for ModelRejectNode {
    type ViewQuery = &'static ModelRejectOutput;

    fn run<'w>(
        &self,
        graph: &mut RenderGraphContext,
        context: &mut RenderContext<'w>,
        output: QueryItem<'w, '_, Self::ViewQuery>,
        world: &'w World,
    ) -> Result<(), NodeRunError> {
        let requests = world.resource::<ModelRejectRequests>();
        let Some(view) = requests.by_render.get(&graph.view_entity()) else {
            return Ok(());
        };
        if output.key != view.key {
            return Ok(());
        }
        let Some(pipeline) = world.get_resource::<ModelRejectPipeline>() else {
            return Ok(());
        };
        let cache = world.resource::<PipelineCache>();
        let Some(render_pipeline) = cache.get_render_pipeline(pipeline.pipeline) else {
            return Ok(());
        };
        let epoch = requests.epoch;
        let encoded = output
            .masks
            .encode_current(epoch, view.key.generation, |destination| {
                let mut pass = context
                    .command_encoder()
                    .begin_render_pass(&RenderPassDescriptor {
                        label: Some("temporal model rejection"),
                        color_attachments: &[Some(RenderPassColorAttachment {
                            view: &destination.default_view,
                            resolve_target: None,
                            depth_slice: None,
                            ops: Operations {
                                load: LoadOp::Clear(Default::default()),
                                store: StoreOp::Store,
                            },
                        })],
                        ..default()
                    });
                pass.set_pipeline(render_pipeline);
                if view.full_view {
                    pass.draw(0..3, 0..1);
                } else {
                    for rect in &view.rects {
                        let width = rect.max.x.saturating_sub(rect.min.x);
                        let height = rect.max.y.saturating_sub(rect.min.y);
                        if width == 0 || height == 0 {
                            continue;
                        }
                        pass.set_scissor_rect(rect.min.x, rect.min.y, width, height);
                        pass.draw(0..3, 0..1);
                    }
                }
                drop(pass);
                true
            });
        if encoded {
            debug_assert!(output.current_mask(epoch, view.key.generation).is_some());
        }
        Ok(())
    }
}

/// Private child plugin: registered from [`super::AaPlugin`].
pub(super) struct ModelRejectionPlugin;

impl Plugin for ModelRejectionPlugin {
    fn build(&self, app: &mut App) {
        let Some(render_app) = app.get_sub_app_mut(RenderApp) else {
            return;
        };
        render_app
            .init_resource::<ModelRejectRequests>()
            .add_systems(
                ExtractSchedule,
                extract_model_rejections
                    .after(depth::extract_depth_requests)
                    .after(extract_resource::<RetainedTemporalSource>),
            )
            .add_systems(
                Render,
                prepare_model_reject_outputs
                    .after(depth::prepare_depth_textures)
                    .in_set(RenderSystems::PrepareBindGroups),
            )
            .add_systems(RenderStartup, init_model_reject_pipeline)
            .add_render_graph_node::<ViewNodeRunner<ModelRejectNode>>(Core3d, ModelRejectLabel)
            .add_render_graph_edges(
                Core3d,
                (
                    Node3d::MainOpaquePass,
                    ModelRejectLabel,
                    TemporalOpaqueSnapshotLabel,
                ),
            );
    }
}

#[cfg(test)]
pub(super) fn gpu_probe(
    world: &World,
    render: Entity,
    epoch: u64,
    generation: u64,
) -> Option<super::gpu_validation::TextureProbe> {
    let output = world.get::<ModelRejectOutput>(render)?;
    Some(super::gpu_validation::TextureProbe {
        key: output.key(),
        current: output
            .masks
            .current(epoch, generation)
            .map(|texture| texture.texture.clone()),
        previous: output
            .masks
            .previous(epoch, generation)
            .map(|texture| texture.texture.clone()),
    })
}

#[cfg(test)]
pub(super) fn gpu_pipeline_ready(world: &World) -> bool {
    world
        .get_resource::<ModelRejectPipeline>()
        .and_then(|pipeline| {
            world
                .get_resource::<PipelineCache>()?
                .get_render_pipeline(pipeline.pipeline)
        })
        .is_some()
}

#[cfg(test)]
pub(super) fn gpu_pipeline_diagnostics(world: &World) -> Vec<String> {
    let Some(pipeline) = world.get_resource::<ModelRejectPipeline>() else {
        return vec!["rejection pipeline: resource absent".into()];
    };
    let cache = world.resource::<PipelineCache>();
    vec![format!(
        "rejection pipeline: {:?}",
        cache.get_render_pipeline_state(pipeline.pipeline)
    )]
}

#[cfg(test)]
mod tests;
