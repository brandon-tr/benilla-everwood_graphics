// Bevy-derived reversible-tonemap resolve with independently owned reconstructed history.
// Tonemap/reverse-tonemap and Playdead YCoCg/variance-clipping helpers are adapted from
// bevy_anti_alias 0.18.1 taa.wgsl (MIT), including Playdead's MIT-licensed helpers.
// Upstream: https://github.com/bevyengine/bevy/blob/v0.18.1/crates/bevy_anti_alias/src/taa/taa.wgsl
// Playdead: https://github.com/playdeadgames/temporal/blob/master/Assets/Shaders/TemporalReprojection.shader

struct Motion {
    current_view_from_clip: mat4x4<f32>,
    previous_clip_from_current_view: mat4x4<f32>,
    previous_view_from_clip: mat4x4<f32>,
    extent_and_flags: vec4<f32>,
    raster_jitter_delta: vec4<f32>,
};

@group(0) @binding(0) var current_scene: texture_2d<f32>;
@group(0) @binding(1) var previous_history: texture_2d<f32>;
@group(0) @binding(2) var motion_candidate: texture_2d<f32>;
@group(0) @binding(3) var current_depth: texture_2d<f32>;
@group(0) @binding(4) var previous_depth: texture_2d<f32>;
@group(0) @binding(5) var current_reactive: texture_2d<f32>;
@group(0) @binding(6) var previous_reactive: texture_2d<f32>;
@group(0) @binding(7) var current_model: texture_2d<f32>;
@group(0) @binding(8) var previous_model: texture_2d<f32>;
@group(0) @binding(9) var<uniform> params: Motion;

struct Output {
    @location(0) scene: vec4<f32>,
    @location(1) history: vec4<f32>,
};

const MIN_HISTORY_BLEND_RATE: f32 = 0.015;
const DEFAULT_HISTORY_BLEND_RATE: f32 = 0.1;

fn finite(x: f32) -> bool { return x == x && abs(x) < 3.402823e38; }
fn finite3(v: vec3<f32>) -> bool { return finite(v.x) && finite(v.y) && finite(v.z); }
fn finite4(v: vec4<f32>) -> bool { return finite(v.x) && finite(v.y) && finite(v.z) && finite(v.w); }
fn max3(v: vec3<f32>) -> f32 { return max(v.r, max(v.g, v.b)); }
fn safe_source(c: vec4<f32>) -> bool { return finite4(c) && c.r >= 0.0 && c.g >= 0.0 && c.b >= 0.0; }
fn tonemap(c: vec3<f32>) -> vec3<f32> {
    let denom = max3(c) + 1.0;
    let mapped = c / denom;
    if (!finite(denom) || abs(denom) <= 1e-8 || !finite3(mapped)) {
        return vec3(-1.0);
    }
    return mapped;
}
fn reverse_tonemap(c: vec3<f32>) -> vec3<f32> {
    let denom = 1.0 - max3(c);
    let mapped = c / denom;
    if (!finite(denom) || abs(denom) <= 1e-8 || !finite3(mapped) || !finite3(c)) {
        return vec3(-1.0);
    }
    return mapped;
}
fn valid_tonemapped(c: vec3<f32>) -> bool {
    return finite3(c) && c.r >= 0.0 && c.g >= 0.0 && c.b >= 0.0
        && c.r < 1.0 && c.g < 1.0 && c.b < 1.0;
}
fn RGB_to_YCoCg(rgb: vec3<f32>) -> vec3<f32> {
    return vec3((rgb.r / 4.0) + (rgb.g / 2.0) + (rgb.b / 4.0),
                (rgb.r / 2.0) - (rgb.b / 2.0),
                (-rgb.r / 4.0) + (rgb.g / 2.0) - (rgb.b / 4.0));
}
fn YCoCg_to_RGB(v: vec3<f32>) -> vec3<f32> {
    return saturate(vec3(v.x + v.y - v.z, v.x + v.z, v.x - v.y - v.z));
}
fn clip_towards_aabb_center(history_color: vec3<f32>, aabb_min: vec3<f32>, aabb_max: vec3<f32>) -> vec3<f32> {
    let p = 0.5 * (aabb_max + aabb_min);
    let e = 0.5 * (aabb_max - aabb_min) + vec3(0.00000001);
    let v = history_color - p;
    let unit = v / e;
    let ma = max3(abs(unit));
    if (ma > 1.0) { return p + v / ma; }
    return history_color;
}
fn reconstruct(uv: vec2<f32>, depth: f32, view_from_clip: mat4x4<f32>) -> vec4<f32> {
    let ndc = vec4(2.0 * uv.x - 1.0, 1.0 - 2.0 * uv.y, depth, 1.0);
    let h = view_from_clip * ndc;
    if (!finite4(h) || abs(h.w) < 1e-8) { return vec4(0.0); }
    let v = h.xyz / h.w;
    if (!finite3(v)) { return vec4(0.0); }
    return vec4(v, 1.0);
}
fn mask_rejects(t: texture_2d<f32>, pixel: vec2<i32>) -> bool {
    let v = textureLoad(t, pixel, 0).x;
    return !finite(v) || v != 0.0;
}
fn dims_match(t: texture_2d<f32>, extent: vec2<f32>) -> bool {
    let s = textureDimensions(t);
    return s.x == u32(extent.x) && s.y == u32(extent.y);
}
fn support_ok(raw_uv: vec2<f32>, expected_distance: f32, extent: vec2<f32>, previous_view_from_clip: mat4x4<f32>) -> bool {
    if (any(raw_uv < vec2(0.0)) || any(raw_uv >= vec2(1.0))) { return false; }
    let base = vec2<i32>(floor(raw_uv * extent - vec2<f32>(0.5)));
    let max_coord = vec2<i32>(extent) - vec2<i32>(1);
    for (var dy = 0; dy <= 1; dy = dy + 1) {
        for (var dx = 0; dx <= 1; dx = dx + 1) {
            let tap = clamp(base + vec2<i32>(dx, dy), vec2<i32>(0), max_coord);
            if (mask_rejects(previous_reactive, tap) || mask_rejects(previous_model, tap)) { return false; }
            let d = textureLoad(previous_depth, tap, 0).x;
            if (!finite(d) || d <= 0.0 || d > 1.0) { return false; }
            let uv = (vec2<f32>(tap) + vec2(0.5)) / extent;
            let view = reconstruct(uv, d, previous_view_from_clip);
            if (view.w < 0.5) { return false; }
            let actual = -view.z;
            if (!finite(actual) || actual <= 0.0) { return false; }
            let tolerance = max(0.02, 0.005 * expected_distance);
            if (abs(actual - expected_distance) > tolerance) { return false; }
        }
    }
    return true;
}

@vertex
fn vertex(@builtin(vertex_index) i: u32) -> @builtin(position) vec4<f32> {
    let x = f32((i << 1u) & 2u);
    let y = f32(i & 2u);
    return vec4(x * 2.0 - 1.0, y * 2.0 - 1.0, 0.0, 1.0);
}

@fragment
fn seed_fragment(@builtin(position) p: vec4<f32>) -> Output {
    let current = textureLoad(current_scene, vec2<i32>(p.xy), 0);
    var out: Output;
    out.scene = current;
    out.history = vec4(0.0, 0.0, 0.0, 0.0);
    if (safe_source(current)) {
        let mapped = tonemap(current.rgb);
        if (valid_tonemapped(mapped)) {
            out.history = vec4(mapped, 1.0);
        }
    }
    return out;
}

fn current_stat_sample(t: texture_2d<f32>, tap: vec2<i32>) -> vec3<f32> {
    let c = textureLoad(t, tap, 0);
    if (!safe_source(c)) { return vec3(3.402823e38); }
    let mapped = tonemap(c.rgb);
    if (!valid_tonemapped(mapped)) { return vec3(3.402823e38); }
    return RGB_to_YCoCg(mapped);
}

@fragment
fn temporal_fragment(@builtin(position) p: vec4<f32>) -> Output {
    let extent = params.extent_and_flags.xy;
    let current = textureLoad(current_scene, vec2<i32>(p.xy), 0);
    var out: Output;
    out.scene = current;
    out.history = vec4(0.0, 0.0, 0.0, 0.0);
    if (extent.x <= 0.0 || extent.y <= 0.0 || params.extent_and_flags.z < 0.5) { return out; }
    if (!safe_source(current)) { return out; }
    let current_tonemapped = tonemap(current.rgb);
    if (!valid_tonemapped(current_tonemapped)) { return out; }
    out.history = vec4(current_tonemapped, 1.0);
    if (!dims_match(current_scene, extent) || !dims_match(previous_history, extent)
        || !dims_match(motion_candidate, extent) || !dims_match(current_reactive, extent)
        || !dims_match(previous_reactive, extent) || !dims_match(current_model, extent)
        || !dims_match(previous_model, extent) || !dims_match(current_depth, extent)
        || !dims_match(previous_depth, extent)) { return out; }

    let pixel = vec2<i32>(p.xy);
    let max_coord = vec2<i32>(extent) - vec2<i32>(1);
    let current_uv = p.xy / extent;
    let depth = textureLoad(current_depth, pixel, 0).x;
    if (!finite(depth) || depth <= 0.0 || depth > 1.0) { return out; }
    let candidate = textureLoad(motion_candidate, pixel, 0);
    if (!finite(candidate.w) || candidate.w < 0.5) { return out; }
    if (mask_rejects(current_reactive, pixel) || mask_rejects(current_model, pixel)) { return out; }

    let current_view = reconstruct(current_uv, depth, params.current_view_from_clip);
    if (current_view.w < 0.5) { return out; }
    let previous_h = params.previous_clip_from_current_view * current_view;
    if (!finite4(previous_h) || previous_h.w <= 1e-8) { return out; }
    let previous_ndc = previous_h.xyz / previous_h.w;
    if (!finite3(previous_ndc) || previous_ndc.z < 0.0 || previous_ndc.z > 1.0) { return out; }
    let previous_raw_uv = vec2(0.5 * previous_ndc.x + 0.5, 0.5 - 0.5 * previous_ndc.y);
    if (!finite(previous_raw_uv.x) || !finite(previous_raw_uv.y)) { return out; }
    let previous_view = reconstruct(previous_raw_uv, previous_ndc.z, params.previous_view_from_clip);
    if (previous_view.w < 0.5) { return out; }
    let expected_distance = -previous_view.z;
    if (!finite(expected_distance) || expected_distance <= 0.0) { return out; }

    let unjittered_motion = current_uv - previous_raw_uv - params.raster_jitter_delta.xy;
    let history_uv = current_uv - unjittered_motion;
    if (!finite(unjittered_motion.x) || !finite(unjittered_motion.y)
        || !finite(history_uv.x) || !finite(history_uv.y)) { return out; }
    if (any(history_uv < vec2(0.0)) || any(history_uv >= vec2(1.0))) { return out; }
    let history_pixel = history_uv * extent - vec2(0.5);
    let base = vec2<i32>(floor(history_pixel));
    let fraction = history_pixel - floor(history_pixel);

    var history_rgb = vec3(0.0);
    var weight_sum = 0.0;
    var min_confidence = 3.402823e38;
    for (var dy = 0; dy <= 1; dy = dy + 1) {
        for (var dx = 0; dx <= 1; dx = dx + 1) {
            let tap = clamp(base + vec2<i32>(dx, dy), vec2<i32>(0), max_coord);
            let tap_uv = (vec2<f32>(tap) + vec2(0.5)) / extent;
            let raw_tap_uv = tap_uv + (previous_raw_uv - history_uv);
            if (!support_ok(raw_tap_uv, expected_distance, extent, params.previous_view_from_clip)) { return out; }
            let sample = textureLoad(previous_history, tap, 0);
            if (!finite4(sample) || !finite(sample.w) || sample.w <= 0.0
                || !valid_tonemapped(sample.rgb)) { return out; }
            let wx = select(1.0 - fraction.x, fraction.x, dx == 1);
            let wy = select(1.0 - fraction.y, fraction.y, dy == 1);
            let weight = wx * wy;
            history_rgb += sample.rgb * weight;
            weight_sum += weight;
            min_confidence = min(min_confidence, sample.w);
        }
    }
    if (weight_sum <= 1e-8) { return out; }
    history_rgb /= weight_sum;

    var aabb_min = vec3(3.402823e38);
    var aabb_max = vec3(-3.402823e38);
    var mean = vec3(0.0);
    for (var dy = -1; dy <= 1; dy = dy + 1) {
        for (var dx = -1; dx <= 1; dx = dx + 1) {
            let tap = clamp(pixel + vec2<i32>(dx, dy), vec2<i32>(0), max_coord);
            let c = current_stat_sample(current_scene, tap);
            if (c.x > 1.0e30) { return out; }
            mean += c;
            aabb_min = min(aabb_min, c);
            aabb_max = max(aabb_max, c);
        }
    }
    mean /= 9.0;
    var variance = vec3(0.0);
    for (var dy = -1; dy <= 1; dy = dy + 1) {
        for (var dx = -1; dx <= 1; dx = dx + 1) {
            let tap = clamp(pixel + vec2<i32>(dx, dy), vec2<i32>(0), max_coord);
            let c = current_stat_sample(current_scene, tap);
            if (c.x > 1.0e30) { return out; }
            variance += (c - mean) * (c - mean);
        }
    }
    variance /= 9.0;
    let sigma = sqrt(variance);
    aabb_min = max(aabb_min, mean - sigma);
    aabb_max = min(aabb_max, mean + sigma);

    let history_ycocg = RGB_to_YCoCg(history_rgb);
    var clipped = clip_towards_aabb_center(history_ycocg, aabb_min, aabb_max);
    clipped = YCoCg_to_RGB(clipped);
    var confidence = min_confidence;
    if (max(abs(unjittered_motion.x * extent.x), abs(unjittered_motion.y * extent.y)) < 0.01) {
        confidence = min(min_confidence + 10.0, 1.0 / MIN_HISTORY_BLEND_RATE);
    } else {
        confidence = 1.0;
    }
    if (!finite(confidence) || confidence <= 0.0) { return out; }
    let factor = clamp(1.0 / confidence, MIN_HISTORY_BLEND_RATE, DEFAULT_HISTORY_BLEND_RATE);
    let mixed = mix(clipped, current_tonemapped, factor);
    if (!finite3(mixed) || !finite3(clipped) || !valid_tonemapped(mixed)) { return out; }
    let reconstructed = reverse_tonemap(mixed);
    if (any(reconstructed < vec3(0.0)) || !finite3(reconstructed)) { return out; }
    out.scene = vec4(reconstructed, current.a);
    out.history = vec4(mixed, confidence);
    return out;
}
