//! Retained tree-wind epoch tracking for the native prepass.

use bevy::anti_alias::taa::TemporalAntiAliasing;
use bevy::core_pipeline::prepass::{DeferredPrepass, DepthPrepass, MotionVectorPrepass};
use bevy::diagnostic::FrameCount;
use bevy::prelude::*;
use bevy::render::camera::ExtractedCamera;
use bevy::render::render_resource::{
    BindGroup, BindGroupEntries, DynamicUniformBuffer, PipelineCache, ShaderType,
};
use bevy::render::renderer::{RenderDevice, RenderQueue};
use bevy::render::view::{ExtractedView, RetainedViewEntity};

use crate::lighting::{GxRetainedWindFrame, SharedLightBuffer};

use super::{GxPipelines, GxWorld, StaticGxMirror, StaticGxView};

#[derive(ShaderType, Clone, Copy, Debug, Default)]
pub(super) struct GxPreviousWindData {
    pub wind_a: Vec4,
    pub wind_b: Vec4,
    /// x = valid previous rows; y = source present.
    pub flags: Vec4,
}

#[derive(Resource, Default)]
pub(super) struct GxPreviousWindUniforms {
    pub buffer: DynamicUniformBuffer<GxPreviousWindData>,
}

#[derive(Component, Clone, Copy)]
pub(super) struct GxPreparedWind {
    data: GxPreviousWindData,
    wind_ready: bool,
}

#[derive(Component)]
pub(super) struct GxPreviousWindOffset(pub u32);

#[derive(Component)]
pub(super) struct GxPreviousWindBind {
    pub bind: BindGroup,
    pub wind_ready: bool,
}

#[derive(Clone, Copy)]
struct GxViewWind {
    last_frame: u32,
    target: UVec2,
    buffer: Option<bevy::render::render_resource::BufferId>,
    current_a: [f32; 4],
    current_b: [f32; 4],
    previous_a: [f32; 4],
    previous_b: [f32; 4],
    valid: bool,
}

#[derive(Resource, Default)]
pub(super) struct GxRetainedWindHistory {
    views: bevy::platform::collections::HashMap<RetainedViewEntity, GxViewWind>,
}

fn row(v: [f32; 4]) -> Vec4 {
    Vec4::from_array(v)
}

fn wind_needed(gx: &GxWorld) -> bool {
    let mut needed = false;
    gx.visit_temporal_wind(|region| {
        needed = true;
        let _ = region;
        false
    });
    needed
}

/// Pure epoch transition. `buffer_changed` is supplied by the caller so CPU tests
/// need no GPU buffer identity.
fn advance_wind_epoch(
    old: Option<&GxViewWind>,
    frame: u32,
    target: UVec2,
    buffer_changed: bool,
    current: ([f32; 4], [f32; 4]),
    caller_reset: bool,
) -> GxViewWind {
    if let Some(old) = old {
        if old.last_frame == frame {
            let mut same = *old;
            if caller_reset {
                same.valid = false;
            }
            return same;
        }
    }
    let consecutive = old.is_some_and(|old| {
        old.last_frame.wrapping_add(1) == frame
            && old.target == target
            && !buffer_changed
            && !caller_reset
    });
    match (consecutive, old) {
        (true, Some(old)) => GxViewWind {
            last_frame: frame,
            target,
            buffer: old.buffer,
            current_a: current.0,
            current_b: current.1,
            previous_a: old.current_a,
            previous_b: old.current_b,
            valid: true,
        },
        _ => GxViewWind {
            last_frame: frame,
            target,
            buffer: old.and_then(|o| o.buffer),
            current_a: current.0,
            current_b: current.1,
            previous_a: current.0,
            previous_b: current.1,
            valid: false,
        },
    }
}

#[allow(clippy::too_many_arguments)]
#[allow(clippy::type_complexity)]
pub(super) fn prepare_retained_wind_epoch(
    mut commands: Commands,
    frame: Res<FrameCount>,
    current: Option<Res<GxRetainedWindFrame>>,
    light_buffer: Option<Res<SharedLightBuffer>>,
    gx: Res<GxWorld>,
    mut history: ResMut<GxRetainedWindHistory>,
    mut views: Query<
        (
            Entity,
            &ExtractedView,
            &mut TemporalAntiAliasing,
            &StaticGxView,
            &Msaa,
            &DepthPrepass,
            &MotionVectorPrepass,
            Has<DeferredPrepass>,
            &ExtractedCamera,
            Option<&bevy::camera::MainPassResolutionOverride>,
        ),
        Without<StaticGxMirror>,
    >,
    mut retired: Query<
        Entity,
        Or<(
            With<GxPreparedWind>,
            With<GxPreviousWindOffset>,
            With<GxPreviousWindBind>,
        )>,
    >,
) {
    let has_wind = wind_needed(&gx);
    let buffer_id = light_buffer.as_ref().map(|b| b.0.id());
    let mut active = bevy::platform::collections::HashSet::<RetainedViewEntity>::default();
    let mut supported_entities = bevy::platform::collections::HashSet::<Entity>::default();
    let rows = current.as_ref().map(|c| (c.rows[4], c.rows[5]));

    for (entity, view, mut taa, _, msaa, _, _, has_deferred, camera, override_) in &mut views {
        if *msaa != Msaa::Off
            || has_deferred
            || override_.is_some()
            || camera.viewport.is_some()
            || camera.physical_target_size.is_none()
        {
            commands
                .entity(entity)
                .remove::<(GxPreparedWind, GxPreviousWindOffset, GxPreviousWindBind)>();
            continue;
        }
        let target = camera.physical_target_size.unwrap();
        if target.x == 0 || target.y == 0 {
            commands
                .entity(entity)
                .remove::<(GxPreparedWind, GxPreviousWindOffset, GxPreviousWindBind)>();
            continue;
        }
        let key = view.retained_view_entity;
        active.insert(key);
        supported_entities.insert(entity);

        let Some(rows) = rows else {
            history.views.remove(&key);
            if has_wind {
                taa.reset = true;
            }
            commands
                .entity(entity)
                .remove::<(GxPreviousWindOffset, GxPreviousWindBind)>();
            commands.entity(entity).insert(GxPreparedWind {
                data: GxPreviousWindData::default(),
                wind_ready: false,
            });
            continue;
        };

        let buffer_changed = history
            .views
            .get(&key)
            .is_some_and(|old| old.buffer != buffer_id);
        let next = advance_wind_epoch(
            history.views.get(&key),
            frame.0,
            target,
            buffer_changed,
            rows,
            taa.reset,
        );
        let duplicate = history
            .views
            .get(&key)
            .is_some_and(|old| old.last_frame == frame.0);
        if !next.valid && !duplicate {
            taa.reset = true;
        }
        if duplicate && taa.reset {
            taa.reset = true;
        }
        let mut stored = next;
        stored.buffer = buffer_id;
        history.views.insert(key, stored);
        commands.entity(entity).insert(GxPreparedWind {
            data: GxPreviousWindData {
                wind_a: row(next.previous_a),
                wind_b: row(next.previous_b),
                flags: Vec4::new(f32::from(u8::from(next.valid)), 1.0, 0.0, 0.0),
            },
            wind_ready: true,
        });
    }

    history.views.retain(|k, _| active.contains(k));
    for entity in &mut retired {
        if !supported_entities.contains(&entity) {
            commands
                .entity(entity)
                .remove::<(GxPreparedWind, GxPreviousWindOffset, GxPreviousWindBind)>();
        }
    }
}

#[allow(clippy::type_complexity)]
pub(super) fn prepare_previous_wind_uniforms(
    mut commands: Commands,
    render_device: Res<RenderDevice>,
    render_queue: Res<RenderQueue>,
    mut uniforms: ResMut<GxPreviousWindUniforms>,
    views: Query<(Entity, &GxPreparedWind), With<super::GxEligibleNativePrepass>>,
    mut ineligible: Query<
        (Entity, Option<&GxPreviousWindOffset>),
        (
            With<GxPreparedWind>,
            Without<super::GxEligibleNativePrepass>,
        ),
    >,
    mut orphaned: Query<Entity, (With<GxPreviousWindOffset>, Without<GxPreparedWind>)>,
) {
    for (entity, offset) in &mut ineligible {
        if offset.is_some() {
            commands
                .entity(entity)
                .remove::<(GxPreviousWindOffset, GxPreviousWindBind)>();
        }
    }
    for entity in &mut orphaned {
        commands
            .entity(entity)
            .remove::<(GxPreviousWindOffset, GxPreviousWindBind)>();
    }
    let count = views.iter().len();
    if count == 0 {
        return;
    }
    let Some(mut writer) = uniforms
        .buffer
        .get_writer(count, &render_device, &render_queue)
    else {
        return;
    };
    for (entity, prepared) in &views {
        let offset = writer.write(&prepared.data);
        commands.entity(entity).insert(GxPreviousWindOffset(offset));
    }
}

#[allow(clippy::type_complexity)]
pub(super) fn prepare_previous_wind_bindings(
    mut commands: Commands,
    render_device: Res<RenderDevice>,
    pipeline_cache: Res<PipelineCache>,
    pipes: Res<GxPipelines>,
    uniforms: Res<GxPreviousWindUniforms>,
    views: Query<
        (Entity, &GxPreparedWind, &GxPreviousWindOffset),
        With<super::GxEligibleNativePrepass>,
    >,
    mut stale: Query<
        Entity,
        (
            With<GxPreviousWindBind>,
            Or<(
                Without<GxPreviousWindOffset>,
                Without<GxPreparedWind>,
                Without<super::GxEligibleNativePrepass>,
            )>,
        ),
    >,
) {
    for entity in &mut stale {
        commands.entity(entity).remove::<GxPreviousWindBind>();
    }
    let Some(binding) = uniforms.buffer.binding() else {
        return;
    };
    let layout = pipeline_cache.get_bind_group_layout(&pipes.previous_wind_layout);
    for (entity, prepared, offset) in &views {
        let bind = render_device.create_bind_group(
            "static_gx_previous_wind",
            &layout,
            &BindGroupEntries::single(binding.clone()),
        );
        commands.entity(entity).insert((
            GxPreviousWindOffset(offset.0),
            GxPreviousWindBind {
                bind,
                wind_ready: prepared.wind_ready,
            },
        ));
    }
}

#[cfg(test)]
mod tests;
