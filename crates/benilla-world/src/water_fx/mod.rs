//! Water foam: the reference's `CWater0Ripple` wade wake, standing ring and step-in splash, as its
//! pool records. A record holds a feet-anchored centre on the liquid surface, a heading, a start
//! size, a growth rate, a lifetime and a peak alpha.
//!
//! A record's geometry is built once, at emission, from the wet liquid cells under its final box,
//! which clips foam at banks. Growth is texgen only: each frame the stencil maps `centre ± size(t)`
//! to UV [0, 1] at the heading, clamped to its transparent border.
//!
//! The driver (`0x5fa760`) emits a wake when translating, a ring when turning or standing, and a
//! full ring on crossing the wade depth (`0x6030c0`).
//!
//! The patch is the liquid mesh's own triangles through the same `clip_from_world`, so the depth
//! test settles its tie with the water; a lift would overshoot onto dry sand on a beach. The
//! reference also lays foam at the MCLQ height, with only a depth bias (`0x68fd0f`):
//! `0.125 × [0x810390]`, D3D `DEPTHBIAS` −1/8192.

mod bob;
mod params;

use bevy::asset::RenderAssetUsages;
use bevy::ecs::entity::EntityHashMap;
use bevy::image::{ImageAddressMode, ImageFilterMode, ImageSampler, ImageSamplerDescriptor};
use bevy::platform::collections::HashMap;
use bevy::prelude::*;
use bevy::render::render_resource::{Extent3d, TextureDimension, TextureFormat};

use benilla_assets::coords::{bevy_to_wow, wow_to_bevy};

use crate::liquid::{FoamPatch, WaterChunkInfo, WaterIndex};
use crate::particles::buffer::{
    begin_effect_frame, EffectBlend, EffectDrawSpec, EffectFog, EffectQuads, EffectVertex,
};
use crate::schedule::WorldStage;
use crate::view::WorldCamera;
use crate::world_unit::{ViewerUnit, WorldUnit};
use benilla_assets::{AssetSet, LooseOpen, WorldAssets};

// MONKEY: stock and plugin emission share the record parameters and drawing math.
use params::{foam_gain, swim_ramp, tread_params, FoamLook, TREAD_INTERVAL};
use params::{foam_params, foam_uv, rand01, record_alpha, record_size, FoamParams, WadeState};
use params::{wake_cooldown, RING_INTERVAL};

/// The ring and wake stencils, loaded raw: the alpha is the shape and the near-black RGB the
/// intensity, as the reference modulates both, so foam recoloured white is about 20× too bright.
const RING_TEXTURE: &str = "xtextures/splash/splash.blp";
const WAKE_TEXTURE: &str = "xtextures/splash/wake.blp";

/// The record pool (`0x68f8b0`, `0x68f9f0`): the active mover allocates from `[0, 32)` and
/// everyone else from `[32, 128)`, so others never evict its records.
const POOL_SIZE: usize = 128;
const SELF_SLOTS: usize = 32;

// MONKEY: external emissions own their capacity and never touch stock records/cooldowns.
const PLUGIN_POOL_SIZE: usize = 96;
// MONKEY: bound the spatial-index/geometry walk even for malformed finite plugin parameters.
const PLUGIN_MAX_EXTENT: f32 = 64.0;

/// MONKEY: foam stencil category; splash uses the ring stencil with caller-supplied dynamics.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WaterFoamKind {
    Ring,
    Wake,
    Splash,
}

/// MONKEY: one caller-authored foam patch on loaded water. Heading is WoW radians; sizes
/// are half-widths in yards, growth is yd/s, lifetime is seconds, peak alpha is 0..1.
/// `subject` selects a unit's liquid claim; `None` accepts either ADT or WMO water.
/// Invalid/nonfinite requests and dry points are dropped. Capacity is independent of stock foam.
/// MONKEY: the final half-width must not exceed 64 yd to bound patch construction.
#[derive(Message, Debug, Clone, Copy)]
pub struct WaterFoamSpawn {
    pub subject: Option<Entity>,
    pub center_bevy: Vec3,
    pub kind: WaterFoamKind,
    pub heading: f32,
    pub start_size: f32,
    pub growth: f32,
    pub lifetime: f32,
    pub peak_alpha: f32,
}

/// The step-in and step-out one-shot depth as a fraction of the unit's own collision height
/// (`0x6030c0`, CMovement+0xb4), latched either way: about 0.81 yd for a human.
const ONESHOT_DEPTH_FRAC: f32 = 0.4;

/// The emission depth gate, `max(2 × collision height, 1.0)`: about 4.06 yd for a human, so a
/// surface swimmer at its 1.52-yd rest depth still foams. The field, `[unit+0x297]`, is
/// CMovement+0xb4, the collision height, not `UNIT_FIELD_BOUNDINGRADIUS`.
const GATE_DEPTH_FRAC: f32 = 2.0;

/// Speed (yd/s) past which a streamed unit counts as translating: its `MOVEMENTFLAGS & 0xf` proxy.
const MOVE_EPSILON: f32 = 0.5;
/// Yaw rate (rad/s) past which a still streamed unit counts as turning: its `& 0x30` proxy.
const TURN_EPSILON: f32 = 0.35;

/// A `CWater0Ripple` pool record: static geometry, with size and alpha derived from `born`.
struct FoamRecord {
    /// Feet position at emission, WoW XY.
    center: [f32; 2],
    /// Texgen heading (WoW radians): the movement direction (wake) or random (ring).
    heading: f32,
    size0: f32,
    /// yd/s.
    growth: f32,
    /// Seconds; the record dies at `born + lifetime`.
    lifetime: f32,
    peak: f32,
    born: f32,
    /// Render category: ring (`splash.blp`) vs wake (`wake.blp`).
    ring: bool,
    /// Static triangles in Bevy space, on the surface, from the wet cells under the final box.
    verts: Vec<Vec3>,
    /// The liquid chunk that hosted the emission; records draw grouped per chunk.
    chunk: Entity,
}

/// Per-unit emitter state: the cooldown cell both kinds share (`unit+0xc78`), so a ring delays the
/// next wake and vice versa; the step-in latch; the motion history for the proxies.
struct UnitFoam {
    last_pos: Option<Vec3>,
    last_yaw: Option<f32>,
    /// Absolute time the next emission is allowed.
    ready: f32,
    /// Step-in latch: currently deeper than the one-shot threshold.
    wading: bool,
    /// MONKEY (swim waves) — the treading ring's OWN deadline, deliberately not the shared cell
    /// above. The reference's one cooldown is a faithful detail (a ring delays a wake and vice
    /// versa); a record the reference does not have must not reach into it, or the Enhanced ring
    /// would start eating the wakes the same pass exists to make more visible.
    tread_ready: f32,
    rng: u32,
    /// Fed this frame; an unfed unit's state retires.
    active: bool,
}

impl UnitFoam {
    fn new(seed: u32) -> Self {
        UnitFoam {
            last_pos: None,
            last_yaw: None,
            ready: 0.0,
            wading: false,
            tread_ready: 0.0,
            rng: seed | 1,
            active: false,
        }
    }
}

/// The record pool, its cursors and the per-unit emitters; the avatar keys `Entity::PLACEHOLDER`.
#[derive(Resource)]
struct WaterFoam {
    pool: Vec<Option<FoamRecord>>,
    self_cursor: usize,
    other_cursor: usize,
    plugin_pool: Vec<Option<FoamRecord>>, // MONKEY: separate bounded plugin pool.
    plugin_cursor: usize,                 // MONKEY: never advances either stock cursor.
    units: EntityHashMap<UnitFoam>,
}

impl Default for WaterFoam {
    fn default() -> Self {
        WaterFoam {
            pool: (0..POOL_SIZE).map(|_| None).collect(),
            self_cursor: 0,
            other_cursor: 0,
            plugin_pool: (0..PLUGIN_POOL_SIZE).map(|_| None).collect(), // MONKEY: plugin capacity.
            plugin_cursor: 0,                                           // MONKEY: plugin cursor.
            units: EntityHashMap::default(),
        }
    }
}

/// The next slot in a partition, its oldest record (counters `0xc7f3b0` self, `0xc81d48` others).
fn alloc_slot(cursor: &mut usize, base: usize, len: usize) -> usize {
    let i = base + *cursor;
    *cursor = (*cursor + 1) % len;
    i
}

/// The two foam stencils. Their draws carry the reference's foam render state (`0x68fae0`):
/// additive, fog off, depth-tested `LEQUAL` without depth write, and a depth bias (`0x68fd0f`).
#[derive(Resource)]
struct FoamAssets {
    ring: Handle<Image>,
    wake: Handle<Image>,
}

/// A rung over the whole water band, not an epsilon over one surface: the reference draws foam
/// once per frame, after all liquid (`0x6816d0` tail-jumps to `0x68fae0` on both branches).
use crate::sky_order::FOAM_BIAS;

/// Loads the stencils raw, clamped and without mips, the reference's sampler state.
fn setup_water_fx(
    mut commands: Commands,
    world_assets: Option<ResMut<WorldAssets>>,
    mut images: ResMut<Assets<Image>>,
) {
    commands.init_resource::<WaterFoam>();
    let Some(mut world_assets) = world_assets else {
        return;
    };
    let Some(ring) = foam_image(&mut world_assets, RING_TEXTURE, &mut images) else {
        return;
    };
    let Some(wake) = foam_image(&mut world_assets, WAKE_TEXTURE, &mut images) else {
        return;
    };
    commands.insert_resource(FoamAssets { ring, wake });
}

/// Decodes a foam stencil with its authored RGBA verbatim, as raw gamma bytes, from the archives
/// alone: `0x68f8b0`'s `TextureCreate` call passes file flag 0 (`0x68f90f`).
fn foam_image(
    world_assets: &mut WorldAssets,
    path: &str,
    images: &mut Assets<Image>,
) -> Option<Handle<Image>> {
    let Some((w, h, rgba)) = world_assets.decode_rgba(path, LooseOpen::Archives) else {
        warn!("water foam: {path} unavailable — no water foam");
        return None;
    };
    let mut image = Image::new(
        Extent3d {
            width: w,
            height: h,
            depth_or_array_layers: 1,
        },
        TextureDimension::D2,
        rgba,
        TextureFormat::Rgba8Unorm,
        RenderAssetUsages::RENDER_WORLD,
    );
    image.sampler = ImageSampler::Descriptor(ImageSamplerDescriptor {
        address_mode_u: ImageAddressMode::ClampToEdge,
        address_mode_v: ImageAddressMode::ClampToEdge,
        mag_filter: ImageFilterMode::Linear,
        min_filter: ImageFilterMode::Linear,
        ..default()
    });
    Some(images.add(image))
}

/// A record's static patch: every wet cell overlapping the final box, cut into the liquid surface's
/// own two triangles in Bevy space, and the chunk that contains the centre.
fn build_patch(
    center: [f32; 2],
    final_size: f32,
    chunks: &[(Entity, &WaterChunkInfo, &FoamPatch)],
) -> Option<(Vec<Vec3>, Entity)> {
    let (lo_x, hi_x) = (center[0] - final_size, center[0] + final_size);
    let (lo_y, hi_y) = (center[1] - final_size, center[1] + final_size);
    let mut verts = Vec::new();
    let mut host: Option<Entity> = None;
    for (entity, info, _foam) in chunks {
        if !info.overlaps(lo_x, hi_x, lo_y, hi_y) {
            continue;
        }
        if host.is_none() && info.contains(center[0], center[1]) {
            host = Some(*entity);
        }
        info.for_each_wet_cell(|[tl, tr, bl, br]| {
            let x = [tl[0], tr[0], bl[0], br[0]];
            let y = [tl[1], tr[1], bl[1], br[1]];
            let (cx0, cx1) = (
                x.iter().copied().fold(f32::MAX, f32::min),
                x.iter().copied().fold(f32::MIN, f32::max),
            );
            let (cy0, cy1) = (
                y.iter().copied().fold(f32::MAX, f32::min),
                y.iter().copied().fold(f32::MIN, f32::max),
            );
            if cx1 < lo_x || cx0 > hi_x || cy1 < lo_y || cy0 > hi_y {
                return;
            }
            // The liquid mesh's own winding: [tl, bl, br] then [tl, br, tr].
            for v in [tl, bl, br, tl, br, tr] {
                verts.push(wow_to_bevy(v));
            }
        });
    }
    let host = host?;
    if verts.is_empty() {
        None
    } else {
        Some((verts, host))
    }
}

/// Build one record's static patch and hand it to the pool — the tail every emission shares.
///
/// Extracted whole (MONKEY (swim waves)) when the treading ring became a second emitter on the
/// same frame: two copies of the candidate walk is two places for the patch clip to drift apart,
/// and the clip is what keeps foam off dry sand.
fn emit_record(
    alloc: &mut dyn FnMut(FoamRecord),
    index: &WaterIndex,
    water: &Query<(Entity, &WaterChunkInfo, &FoamPatch)>,
    center: [f32; 2],
    p: &params::FoamParams,
    heading: f32,
    now: f32,
) {
    let final_size = p.size0 + p.growth * p.lifetime;
    // Candidates for the patch box only — resolved here, on the EMISSION path (past the depth
    // gate and the cooldown), never on the every-frame classify above. `over_box` unions the
    // 1–4 cells a final-size box touches, so the set `build_patch` filters with `overlaps` is
    // the same one the full walk offered it.
    let candidates: Vec<_> = index
        .over_box(
            [center[0] - final_size, center[1] - final_size],
            [center[0] + final_size, center[1] + final_size],
        )
        .into_iter()
        .filter_map(|e| water.get(e).ok())
        .collect();
    if let Some((verts, chunk)) = build_patch(center, final_size, &candidates) {
        alloc(FoamRecord {
            center,
            heading,
            size0: p.size0,
            growth: p.growth,
            lifetime: p.lifetime,
            peak: p.peak,
            born: now,
            ring: p.ring,
            verts,
            chunk,
        });
    }
}

/// Classify + gate + emit for one unit this frame (the driver `0x5fa760`, once per unit per
/// frame; emission paced by the unit's shared cooldown cell).
fn drive_unit(
    foam_state: &mut UnitFoam,
    alloc: &mut dyn FnMut(FoamRecord),
    pos: Vec3,
    state: WadeState,
    scale: f32,
    // The unit's collision height (yd), which both depth lines scale.
    h: f32,
    water: &Query<(Entity, &WaterChunkInfo, &FoamPatch)>,
    index: &WaterIndex,
    now: f32,
    // MONKEY (swim waves) — `WaterQuality >= 1`. The ONE gate on the readability pass; at `false`
    // every line it touches below is the reference's.
    enhanced: bool,
) {
    let gate = (GATE_DEPTH_FRAC * h).max(1.0);
    foam_state.active = true;
    let wow = bevy_to_wow(pos);
    // The surface height from the wet cell underfoot: a chunk's box or top vertex puts a sloped
    // river's wade depth 2 yd out. Through the index, so a dry unit costs one hash miss; a dead
    // entry fails `water.get`, and overlapping surfaces take the first match.
    let Some(surface) = index.over(wow[0], wow[1]).iter().find_map(|&e| {
        let (_, info, _) = water.get(e).ok()?;
        info.surface_z_at(wow[0], wow[1])
    }) else {
        foam_state.wading = false;
        return;
    };
    let depth = surface - wow[2];

    // The one-shot fires on crossing the wade depth either way (`0x6030c0`, latch `[+0x269]`);
    // the driver's mode-0xC9 call also clears the cooldown.
    let wading_now = depth > ONESHOT_DEPTH_FRAC * h;
    let oneshot = wading_now != foam_state.wading;
    foam_state.wading = wading_now;

    // MONKEY (swim waves) — how this record should READ (Enhanced only; `FoamLook::CLASSIC`
    // reproduces the reference exactly). The swim half is the depth law evaluated here rather
    // than a movement flag, because a streamed murloc's flags never reach this crate.
    // Spelled as the constant on the Classic arm rather than as `enhanced: false`: Classic IS the
    // reference look, and stating it that way is what makes "byte-identical" a thing the type
    // says instead of a thing a comment claims. It also keeps the depth law off the Classic
    // frame's hot path entirely — nothing reads it there.
    let look = if enhanced {
        FoamLook {
            enhanced: true,
            swimming: swim_ramp(depth, h) > 0.5,
        }
    } else {
        FoamLook::CLASSIC
    };

    // The treading ring, BEFORE the shared cooldown's early return and on its own cell: a body
    // hanging in deep water is the state the reference's driver renders almost invisible, and it
    // is exactly the state the owner's murloc will be in. Classic touches nothing here — not the
    // timer, not the RNG stream.
    if look.enhanced && look.swimming && !matches!(state, WadeState::Translating { .. }) {
        if now >= foam_state.tread_ready {
            foam_state.tread_ready = now + TREAD_INTERVAL;
            if let Some(p) = tread_params(look, scale, &mut foam_state.rng) {
                let heading = rand01(&mut foam_state.rng) * std::f32::consts::TAU;
                emit_record(alloc, index, water, [wow[0], wow[1]], &p, heading, now);
            }
        }
    }

    if !oneshot && now < foam_state.ready {
        return;
    }
    let Some(p) = foam_params(
        state,
        oneshot,
        scale,
        gate,
        depth,
        look,
        &mut foam_state.rng,
    ) else {
        return;
    };
    let heading = match (p.ring, state) {
        (false, WadeState::Translating { heading, .. }) => heading,
        _ => rand01(&mut foam_state.rng) * std::f32::consts::TAU,
    };
    emit_record(alloc, index, water, [wow[0], wow[1]], &p, heading, now);
    // The shared cooldown cell: the one-shot resets it; a ring pulses on the 400+U[0,50) ms law;
    // a wake re-arms on the distance law (one decal per ~0.625 yd of travel).
    let mut uni = |a: f32, b: f32| a + (b - a) * rand01(&mut foam_state.rng);
    foam_state.ready = if oneshot {
        now
    } else if p.ring {
        now + uni(RING_INTERVAL.0, RING_INTERVAL.1)
    } else {
        let speed = match state {
            WadeState::Translating { speed, .. } => speed,
            _ => 0.0,
        };
        now + wake_cooldown(speed, &mut foam_state.rng)
    };
}

/// Runs the driver for the avatar on its movement flags, and for streamed units on the proxies.
fn emit_water_foam(
    time: Res<Time>,
    quality: Res<benilla_assets::WaterQuality>,
    materials: Option<Res<FoamAssets>>,
    mut foam: ResMut<WaterFoam>,
    viewer: Res<crate::view::Viewer>,
    self_unit: Query<&WorldUnit, With<ViewerUnit>>,
    units: Query<(Entity, &Transform, &WorldUnit), Without<ViewerUnit>>,
    water: Query<(Entity, &WaterChunkInfo, &FoamPatch)>,
    index: Res<WaterIndex>,
) {
    if materials.is_none() {
        return;
    }
    let now = time.elapsed_secs();
    let dt = time.delta_secs().max(1.0e-4);
    // MONKEY (swim waves): Enhanced water gets a readable wake; Classic keeps the reference's.
    let enhanced = quality.0 >= 1;

    for uf in foam.units.values_mut() {
        uf.active = false;
    }

    if !water.is_empty() {
        if let Some(body) = viewer.at {
            let uf = foam
                .units
                .entry(Entity::PLACEHOLDER)
                .or_insert_with(|| UnitFoam::new(0x5EED_F0A5));
            let prev = uf.last_pos.replace(body);
            let vel = prev.map_or(Vec3::ZERO, |p| (body - p) / dt);
            let w = bevy_to_wow(vel);
            let speed = (w[0] * w[0] + w[1] * w[1]).sqrt();
            // Selection off our own movement flags, as `0x5fa760` does; the masks live on `Viewer`.
            let state = if viewer.translating() {
                WadeState::Translating {
                    speed,
                    heading: w[1].atan2(w[0]),
                }
            } else if viewer.turning() {
                WadeState::Turning
            } else {
                WadeState::Standing
            };
            let scale = self_unit.single().map(|u| u.scale).unwrap_or(1.0);
            let h = viewer.height;
            let WaterFoam {
                pool,
                self_cursor,
                units,
                ..
            } = &mut *foam;
            let uf = units.get_mut(&Entity::PLACEHOLDER).expect("inserted above");
            let mut alloc = |rec: FoamRecord| {
                pool[alloc_slot(self_cursor, 0, SELF_SLOTS)] = Some(rec);
            };
            drive_unit(
                uf, &mut alloc, body, state, scale, h, &water, &index, now, enhanced,
            );
        }

        // Streamed units; `Without<ViewerUnit>` excludes the avatar's own wire ghost.
        for (entity, transform, unit) in &units {
            if !unit.wades {
                continue;
            }
            let pos = transform.translation;
            let seed = (entity.to_bits() as u32) ^ 0xA11C_E5ED;
            let uf = foam
                .units
                .entry(entity)
                .or_insert_with(|| UnitFoam::new(seed));
            let prev_pos = uf.last_pos.replace(pos);
            let yaw = transform.rotation.to_euler(EulerRot::YXZ).0;
            let prev_yaw = uf.last_yaw.replace(yaw);
            let vel = prev_pos.map_or(Vec3::ZERO, |p| (pos - p) / dt);
            let w = bevy_to_wow(vel);
            let speed = (w[0] * w[0] + w[1] * w[1]).sqrt();
            let yaw_rate = prev_yaw.map_or(0.0, |p| {
                let mut d = yaw - p;
                while d > std::f32::consts::PI {
                    d -= std::f32::consts::TAU;
                }
                while d < -std::f32::consts::PI {
                    d += std::f32::consts::TAU;
                }
                (d / dt).abs()
            });
            let state = if speed > MOVE_EPSILON {
                WadeState::Translating {
                    speed,
                    heading: w[1].atan2(w[0]),
                }
            } else if yaw_rate > TURN_EPSILON {
                WadeState::Turning
            } else {
                WadeState::Standing
            };
            let h = unit.height;
            let WaterFoam {
                pool,
                other_cursor,
                units,
                ..
            } = &mut *foam;
            let uf = units.get_mut(&entity).expect("inserted above");
            let mut alloc = |rec: FoamRecord| {
                pool[alloc_slot(other_cursor, SELF_SLOTS, POOL_SIZE - SELF_SLOTS)] = Some(rec);
            };
            drive_unit(
                uf, &mut alloc, pos, state, unit.scale, h, &water, &index, now, enhanced,
            );
        }
    }

    // Retire emitter state not fed this frame (unit despawned); its records age out on their own.
    foam.units.retain(|_, uf| uf.active);
}

// MONKEY: validate external requests on the unit's own water, then reuse the stock patch builder.
fn emit_plugin_foam(
    mut requests: MessageReader<WaterFoamSpawn>,
    time: Res<Time>,
    assets: Option<Res<FoamAssets>>,
    mut foam: ResMut<WaterFoam>,
    water: Query<(Entity, &WaterChunkInfo, &FoamPatch)>,
    index: Res<WaterIndex>,
    rooms: Query<&crate::wmo_portal::UnitWmoRoom>,
    placements: crate::liquid::RoomPlacements,
) {
    for request in requests.read() {
        if assets.is_none()
            || !request.center_bevy.is_finite()
            || ![
                request.heading,
                request.start_size,
                request.growth,
                request.lifetime,
                request.peak_alpha,
            ]
            .into_iter()
            .all(f32::is_finite)
            || request.start_size <= 0.0
            || request.growth < 0.0
            || request.lifetime <= 0.0
            || request.peak_alpha <= 0.0
        {
            continue;
        }
        let final_size = request.start_size + request.growth * request.lifetime;
        if !final_size.is_finite() || final_size > PLUGIN_MAX_EXTENT {
            continue;
        }
        let wow = bevy_to_wow(request.center_bevy);
        let claim = request
            .subject
            .map_or(crate::liquid::LiquidClaim::Unknown, |e| {
                crate::liquid::unit_claim(rooms.get(e).ok(), &placements)
            });
        let host = index
            .over(wow[0], wow[1])
            .iter()
            .filter_map(|&e| {
                let (_, info, _) = water.get(e).ok()?;
                let hit = crate::liquid::liquid_at(std::iter::once(info), wow, claim)?;
                if hit.kind.is_fullbright() {
                    return None;
                }
                info.water_z_at(wow[0], wow[1]).map(|z| (e, z))
            })
            .min_by(|a, b| a.1.total_cmp(&b.1));
        if host.is_none() {
            continue;
        }
        let p = FoamParams {
            size0: request.start_size,
            growth: request.growth,
            lifetime: request.lifetime,
            peak: request.peak_alpha.clamp(0.0, 1.0),
            ring: request.kind != WaterFoamKind::Wake,
        };
        let WaterFoam {
            plugin_pool,
            plugin_cursor,
            ..
        } = &mut *foam;
        let mut alloc = |rec| {
            plugin_pool[alloc_slot(plugin_cursor, 0, PLUGIN_POOL_SIZE)] = Some(rec);
        };
        emit_record(
            &mut alloc,
            &index,
            &water,
            [wow[0], wow[1]],
            &p,
            request.heading,
            time.elapsed_secs(),
        );
    }
}

/// Ages the pool and pushes one additive draw per (chunk, category) with live records: static
/// positions, the texgen at `size(t)` and white times the alpha ramp.
fn push_water_foam(
    time: Res<Time>,
    quality: Res<benilla_assets::WaterQuality>,
    assets: Option<Res<FoamAssets>>,
    cam: Query<Entity, With<WorldCamera>>,
    mut foam: ResMut<WaterFoam>,
    mut quads: ResMut<EffectQuads>,
    chunks_alive: Query<(), With<WaterChunkInfo>>,
) {
    let Some(assets) = assets else { return };
    let Ok(cam) = cam.single() else { return };
    let now = time.elapsed_secs();
    // MONKEY (swim waves) — the Enhanced readability gain, on the VERTEX COLOUR (see
    // [`params::foam_gain`] for why it cannot be on the alpha). `1.0` in Classic, so that path's
    // vertex stream is `[1, 1, 1, alpha]` exactly as it has always been.
    let enhanced = quality.0 >= 1;
    let gain = foam_gain(enhanced);

    // MONKEY: age both pools without lending plugin emissions any stock storage.
    let WaterFoam {
        pool, plugin_pool, ..
    } = &mut *foam;
    for slot in pool.iter_mut().chain(plugin_pool.iter_mut()) {
        if slot.as_ref().is_some_and(|r| now - r.born >= r.lifetime) {
            *slot = None;
        }
    }

    // One contiguous draw per (chunk, category).
    let mut groups: HashMap<(Entity, bool), Vec<usize>> = HashMap::default();
    for (i, rec) in foam.pool.iter().chain(foam.plugin_pool.iter()).enumerate() {
        // MONKEY: both pools draw.
        let Some(rec) = rec else { continue };
        if chunks_alive.get(rec.chunk).is_err() {
            continue; // chunk streamed out; the record ages out silently
        }
        groups.entry((rec.chunk, rec.ring)).or_default().push(i);
    }
    for ((_chunk, ring), records) in groups {
        let start = quads.begin();
        let mut centroid = Vec3::ZERO;
        let mut n = 0u32;
        for i in records {
            // MONKEY: combined indices keep stock records first and plugin storage separate.
            let rec = if i < POOL_SIZE {
                &foam.pool[i]
            } else {
                &foam.plugin_pool[i - POOL_SIZE]
            }
            .as_ref()
            .expect("grouped above");
            let size = record_size(rec.size0, rec.growth, rec.born, now);
            let alpha = record_alpha(rec.peak, rec.lifetime, rec.born, now);
            for v in &rec.verts {
                let wow = bevy_to_wow(*v);
                quads.verts.push(EffectVertex {
                    pos: v.to_array(),
                    uv: foam_uv(rec.center, rec.heading, size, [wow[0], wow[1]]),
                    color: [gain, gain, gain, alpha],
                });
                centroid += *v;
                n += 1;
            }
        }
        if n == 0 {
            continue;
        }
        quads.commit_tris(
            start,
            EffectDrawSpec {
                cam,
                texture: if ring {
                    assets.ring.id()
                } else {
                    assets.wake.id()
                },
                blend: EffectBlend::Add,
                // The reference's foam render sets fog off (`0x68fcd0`, `0x68fcd2`, `0x68fcd7`).
                fog: EffectFog::Off,
                // The reference's foam render is its own additive path (`0x68fae0`), not the
                // M2 batch state producer — no GL_LIGHTING on it. Classic keeps that verbatim.
                //
                // MONKEY (swim waves): Enhanced takes the SCENE term instead, and it is the gain
                // above that makes it necessary. At the authored colour the foam is dark enough
                // that an unlit draw is invisible at night anyway; at 3.5× it would be a bright
                // white wake under a midnight sky — the environment-sheet failure mode this lane
                // was built for (`EffectLighting::Scene`'s own doc: the Zul'Gurub waterfall foam
                // as a full-white cutout against shaded terrain). So the two changes ship as one:
                // brighter, and tinted by the world that lights everything around it.
                lighting: if enhanced {
                    crate::particles::buffer::EffectLighting::Scene
                } else {
                    crate::particles::buffer::EffectLighting::None
                },
                anchor: centroid / n as f32,
                bias: FOAM_BIAS,
                // A few ULPs and no slope, as the patch already ties the water. The nonzero
                // constant selects `DECAL_WORLD_CLIP`, the world meshes' matrix, so depths agree.
                raster_bias: crate::sky_order::Rung::FOAM_RASTER,
                raster_slope: crate::sky_order::Rung::FOAM_RASTER_SLOPE,
                cam_relative: false,
                // Never the chunk entity: an entity that owns a registered mesh lets bevy's
                // sorted-phase batcher claim the item and rewrite its `batch_range`, a crash.
                no_depth_test: false,
                main_entity: Entity::PLACEHOLDER,
                light: None,
                clip: None,
            },
        );
    }
}

/// The foam emitter's set: the app's `waterfx` capture fixture moves its wading dummy before it.
#[derive(SystemSet, Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct WaterFoamSet;

/// The water foam plugin.
pub(crate) struct WaterFxPlugin;

impl Plugin for WaterFxPlugin {
    fn build(&self, app: &mut App) {
        app.add_message::<WaterFoamSpawn>() // MONKEY: public external foam ingress.
            .add_systems(Startup, setup_water_fx.after(AssetSet::Open))
            .add_systems(
                Update,
                emit_water_foam
                    .in_set(WaterFoamSet)
                    .in_set(WorldStage::Present),
            )
            // MONKEY: separate emitter runs after stock emission, in the public foam ordering set.
            .add_systems(
                Update,
                emit_plugin_foam
                    .after(emit_water_foam)
                    .in_set(WaterFoamSet)
                    .in_set(WorldStage::Present),
            )
            // The push runs after the frame's clear; emission ran in Update.
            .add_systems(PostUpdate, push_water_foam.after(begin_effect_frame));
        // MONKEY (swim waves) — the bodies' half: bobbing on the ocean's vertex swell. It shares
        // this module's water lookup shape and its depth law but nothing else (it emits no
        // records and draws nothing), so it registers itself rather than being threaded in here.
        bob::register(app);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // MONKEY: test the public message ingress, real wet-cell lookup, and independent allocation.
    #[test]
    fn plugin_spawns_drop_dry_points_and_cannot_evict_or_throttle_stock() {
        let mut app = App::new();
        app.add_message::<WaterFoamSpawn>()
            .init_resource::<Time>()
            .init_resource::<WaterIndex>()
            .init_resource::<WaterFoam>()
            .insert_resource(FoamAssets {
                ring: Handle::default(),
                wake: Handle::default(),
            })
            .add_systems(
                Update,
                (crate::liquid::maintain_water_index, emit_plugin_foam).chain(),
            );
        let info = WaterChunkInfo::new(
            crate::liquid::LiquidSource::AdtChunk,
            benilla_formats::LiquidKind::Still,
            [2, 2],
            vec![
                [0.0, 0.0, 5.0],
                [10.0, 0.0, 5.0],
                [0.0, 10.0, 5.0],
                [10.0, 10.0, 5.0],
            ],
            vec![true],
        );
        let chunk = app.world_mut().spawn((info, FoamPatch)).id();
        // MONKEY: a dry hole inside an indexed chunk and magma are also non-water controls.
        for (offset, kind, wet) in [
            (20.0, benilla_formats::LiquidKind::Still, false),
            (30.0, benilla_formats::LiquidKind::Magma, true),
        ] {
            let positions = (0..3)
                .flat_map(|j| (0..3).map(move |i| [offset + i as f32 * 5.0, j as f32 * 5.0, 5.0]))
                .collect();
            app.world_mut().spawn((
                WaterChunkInfo::new(
                    crate::liquid::LiquidSource::AdtChunk,
                    kind,
                    [3, 3],
                    positions,
                    if wet {
                        vec![true; 4]
                    } else {
                        vec![true, false, false, true]
                    },
                ),
                FoamPatch,
            ));
        }
        let subject = app.world_mut().spawn_empty().id();
        {
            let mut foam = app.world_mut().resource_mut::<WaterFoam>();
            foam.self_cursor = 3;
            foam.other_cursor = 5;
            let mut unit = UnitFoam::new(42);
            unit.ready = 123.0;
            foam.units.insert(subject, unit);
            for (i, slot) in foam.pool.iter_mut().enumerate() {
                *slot = Some(FoamRecord {
                    center: [i as f32, 0.0],
                    heading: 0.0,
                    size0: 1.0,
                    growth: 0.0,
                    lifetime: 100.0,
                    peak: 0.25,
                    born: 0.0,
                    ring: true,
                    verts: vec![Vec3::ONE],
                    chunk,
                });
            }
        }
        let request = WaterFoamSpawn {
            subject: Some(subject),
            center_bevy: wow_to_bevy([2.0, 2.0, 4.9]),
            kind: WaterFoamKind::Ring,
            heading: 0.5,
            start_size: 0.5,
            growth: 0.25,
            lifetime: 1.0,
            peak_alpha: 0.75,
        };
        app.world_mut().write_message(WaterFoamSpawn {
            center_bevy: wow_to_bevy([50.0, 50.0, 5.0]),
            ..request
        });
        // MONKEY: dry cells, non-water liquids and invalid requests consume no plugin slots.
        for x in [27.0, 32.0] {
            app.world_mut().write_message(WaterFoamSpawn {
                subject: None,
                center_bevy: wow_to_bevy([x, 2.0, 5.0]),
                ..request
            });
        }
        for invalid in [
            WaterFoamSpawn {
                heading: f32::NAN,
                ..request
            },
            WaterFoamSpawn {
                start_size: PLUGIN_MAX_EXTENT + 1.0,
                ..request
            },
            WaterFoamSpawn {
                lifetime: 0.0,
                ..request
            },
        ] {
            app.world_mut().write_message(invalid);
        }
        app.update();
        assert_eq!(app.world().resource::<WaterFoam>().plugin_cursor, 0);
        assert!(app
            .world()
            .resource::<WaterFoam>()
            .plugin_pool
            .iter()
            .all(Option::is_none));
        for i in 0..PLUGIN_POOL_SIZE + 5 {
            app.world_mut().write_message(WaterFoamSpawn {
                heading: i as f32,
                kind: [
                    WaterFoamKind::Ring,
                    WaterFoamKind::Wake,
                    WaterFoamKind::Splash,
                ][i % 3],
                ..request
            });
        }
        app.update();
        let foam = app.world().resource::<WaterFoam>();
        assert_eq!(foam.plugin_pool.len(), PLUGIN_POOL_SIZE);
        assert_eq!(foam.plugin_pool.iter().flatten().count(), PLUGIN_POOL_SIZE);
        assert_eq!(foam.plugin_cursor, 5);
        assert_eq!(
            foam.plugin_pool[0].as_ref().unwrap().heading,
            PLUGIN_POOL_SIZE as f32
        );
        assert!(!foam.plugin_pool[1].as_ref().unwrap().ring);
        assert!(foam.plugin_pool[2].as_ref().unwrap().ring);
        assert_eq!((foam.self_cursor, foam.other_cursor), (3, 5));
        assert_eq!(foam.units[&subject].ready, 123.0);
        assert_eq!(foam.units.len(), 1);
        for (i, rec) in foam.pool.iter().enumerate() {
            let rec = rec.as_ref().unwrap();
            assert_eq!(rec.center, [i as f32, 0.0]);
            assert_eq!(rec.verts, vec![Vec3::ONE]);
            assert_eq!(rec.peak, 0.25);
        }
    }

    /// An empty or stale index means no foam anywhere, which no helper-level test catches.
    #[test]
    fn the_emitter_finds_its_water_through_the_index() {
        let mut app = App::new();
        app
            // CLASSIC deliberately: this fence pins the REFERENCE emitter's record count, and
            // MONKEY (swim waves)'s Enhanced pass adds a second record (the treading ring) for
            // exactly the body this fixture spawns. The Enhanced count has its own test below.
            .insert_resource(benilla_assets::WaterQuality(0))
            .init_resource::<Time>()
            .init_resource::<WaterIndex>()
            .init_resource::<WaterFoam>()
            .init_resource::<crate::view::Viewer>()
            .insert_resource(FoamAssets {
                ring: Handle::default(),
                wake: Handle::default(),
            })
            .add_systems(
                Update,
                (crate::liquid::maintain_water_index, emit_water_foam).chain(),
            );

        // A 3×3-vertex all-wet grid over [0,10]², surface z = 5.
        let mut positions = Vec::new();
        for j in 0..3 {
            for i in 0..3 {
                positions.push([i as f32 * 5.0, j as f32 * 5.0, 5.0]);
            }
        }
        let info = WaterChunkInfo::new(
            crate::liquid::LiquidSource::AdtChunk,
            benilla_formats::LiquidKind::Still,
            [3, 3],
            positions,
            vec![true; 4],
        );
        app.world_mut().spawn((info, FoamPatch));

        let unit = || WorldUnit {
            wades: true,
            scale: 1.0,
            height: 2.0,
            bound: None,
        };
        // Feet 1.5 yd under, past the 0.4·h step-in line: the one-shot fires on the first frame.
        app.world_mut().spawn((
            Transform::from_translation(wow_to_bevy([2.0, 2.0, 3.5])),
            unit(),
        ));
        // The dry control, far outside every indexed cell.
        app.world_mut().spawn((
            Transform::from_translation(wow_to_bevy([500.0, 500.0, 3.5])),
            unit(),
        ));

        app.update();
        let foam = app.world().resource::<WaterFoam>();
        let live: Vec<usize> = foam
            .pool
            .iter()
            .enumerate()
            .filter_map(|(i, r)| r.as_ref().map(|_| i))
            .collect();
        assert_eq!(live.len(), 1, "one record: the wader's, not the dry unit's");
        assert!(
            live[0] >= SELF_SLOTS,
            "a streamed unit allocates from the other partition"
        );
    }

    /// MONKEY (swim waves) — **Enhanced adds the treading ring, and Classic does not.** Same
    /// world, same body, same seed: the only difference is [`benilla_assets::WaterQuality`], and
    /// the swimmer that emits one reference record at 0 emits that record plus a ring at 1.
    ///
    /// The ring has to come out of a live emitter and not just [`params::tread_params`], because
    /// what it actually has to clear is the shared-cooldown early return the reference's driver
    /// takes on almost every frame — the whole reason it carries its own cadence cell.
    #[test]
    fn enhanced_adds_a_treading_ring_for_a_swimmer() {
        let live = |quality: u8| {
            let mut app = App::new();
            app.insert_resource(benilla_assets::WaterQuality(quality))
                .init_resource::<Time>()
                .init_resource::<WaterIndex>()
                .init_resource::<WaterFoam>()
                .init_resource::<crate::view::Viewer>()
                .insert_resource(FoamAssets {
                    ring: Handle::default(),
                    wake: Handle::default(),
                })
                .add_systems(
                    Update,
                    (crate::liquid::maintain_water_index, emit_water_foam).chain(),
                );
            let mut positions = Vec::new();
            for j in 0..3 {
                for i in 0..3 {
                    positions.push([i as f32 * 5.0, j as f32 * 5.0, 5.0]);
                }
            }
            app.world_mut().spawn((
                WaterChunkInfo::new(
                    crate::liquid::LiquidSource::AdtChunk,
                    benilla_formats::LiquidKind::Ocean,
                    [3, 3],
                    positions,
                    vec![true; 4],
                ),
                FoamPatch,
            ));
            // Feet 1.6 yd under a 2.0-yd body: past the 0.78·h band top, so it is SWIMMING, and
            // standing still, so it is treading. Inside the 2·h = 4 yd emission gate either way.
            app.world_mut().spawn((
                Transform::from_translation(wow_to_bevy([2.0, 2.0, 3.4])),
                WorldUnit {
                    wades: true,
                    scale: 1.0,
                    height: 2.0,
                    bound: None,
                },
            ));
            app.update();
            app.world()
                .resource::<WaterFoam>()
                .pool
                .iter()
                .filter(|r| r.is_some())
                .count()
        };
        assert_eq!(live(0), 1, "Classic: the reference's step-in ring, alone");
        assert_eq!(live(1), 2, "Enhanced: + the treading ring");
    }

    /// Pool partitioning: the self cursor wraps within [0, 32), others within [32, 128) —
    /// eviction replaces the oldest of the same partition, never crosses it.
    #[test]
    fn pool_partitions_and_evicts() {
        let mut foam = WaterFoam::default();
        for _ in 0..SELF_SLOTS + 3 {
            let i = alloc_slot(&mut foam.self_cursor, 0, SELF_SLOTS);
            assert!(i < SELF_SLOTS);
        }
        assert_eq!(foam.self_cursor, 3);
        for _ in 0..(POOL_SIZE - SELF_SLOTS) + 5 {
            let i = alloc_slot(&mut foam.other_cursor, SELF_SLOTS, POOL_SIZE - SELF_SLOTS);
            assert!((SELF_SLOTS..POOL_SIZE).contains(&i));
        }
        assert_eq!(foam.other_cursor, 5);
    }

    #[test]
    fn patch_clips_to_wet_cells() {
        // Four 5-yd cells over `[0,10]²`; only (0,0) is wet, so the box's far half is dry ground.
        let mut positions = Vec::new();
        for j in 0..3 {
            for i in 0..3 {
                positions.push([i as f32 * 5.0, j as f32 * 5.0, 5.0]);
            }
        }
        let info = WaterChunkInfo::new(
            crate::liquid::LiquidSource::AdtChunk,
            benilla_formats::LiquidKind::Still,
            [3, 3],
            positions,
            vec![true, false, false, false],
        );
        let patch = FoamPatch;
        let chunks = vec![(Entity::PLACEHOLDER, &info, &patch)];
        let (verts, _) = build_patch([2.0, 2.0], 1.5, &chunks).unwrap();
        assert_eq!(verts.len(), 6, "only the one wet cell overlaps");
        let wow = bevy_to_wow(verts[0]);
        // On the plane: a lift would overshoot onto dry ground at every shoreline.
        assert!(
            (wow[2] - 5.0).abs() < 1e-4,
            "the patch sits exactly on the liquid surface, unlifted"
        );
        assert!(
            build_patch([50.0, 50.0], 1.5, &chunks).is_none(),
            "dry ⇒ none"
        );
    }
}
