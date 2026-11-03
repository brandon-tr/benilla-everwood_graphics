// Per-view fused camera-motion candidate for stationary opaque surfaces. The output is NOT object
// or deformation motion, confidence, or disocclusion acceptance; a later stage must authorize reuse.
// The history gate is evaluated here at full precision before the result is stored in RGBA16F.

struct Motion {
    current_view_from_clip: mat4x4<f32>,
    previous_clip_from_current_view: mat4x4<f32>,
    previous_view_from_clip: mat4x4<f32>,
    // xy = physical extent, z = previous-frame-usable, w = reserved zero.
    extent_and_flags: vec4<f32>,
    // xy = previous jitter UV minus current jitter UV; zw = zero.
    raster_jitter_delta: vec4<f32>,
};

@group(0) @binding(0) var current_depth: texture_2d<f32>;
@group(0) @binding(1) var<uniform> params: Motion;
@group(0) @binding(2) var previous_depth: texture_2d<f32>;
@group(0) @binding(3) var current_reactive_mask: texture_2d<f32>;
@group(0) @binding(4) var previous_reactive_mask: texture_2d<f32>;
@group(0) @binding(5) var current_model_mask: texture_2d<f32>;
@group(0) @binding(6) var previous_model_mask: texture_2d<f32>;

fn finite(x: f32) -> bool {
    return x == x && abs(x) < 3.402823e38;
}

fn finite3(v: vec3<f32>) -> bool {
    return finite(v.x) && finite(v.y) && finite(v.z);
}

fn finite4(v: vec4<f32>) -> bool {
    return finite(v.x) && finite(v.y) && finite(v.z) && finite(v.w);
}

fn dimensions_match(texture: texture_2d<f32>, extent: vec2<f32>) -> bool {
    let size = textureDimensions(texture);
    return size.x == u32(extent.x) && size.y == u32(extent.y);
}

fn mask_rejects(texture: texture_2d<f32>, pixel: vec2<i32>) -> bool {
    let value = textureLoad(texture, pixel, 0).x;
    return !finite(value) || value != 0.0;
}

fn reconstruct_view(uv: vec2<f32>, depth: f32, view_from_clip: mat4x4<f32>) -> vec4<f32> {
    let ndc = vec4<f32>(2.0 * uv.x - 1.0, 1.0 - 2.0 * uv.y, depth, 1.0);
    let homogeneous = view_from_clip * ndc;
    if (!finite4(homogeneous) || abs(homogeneous.w) < 1e-8) {
        return vec4<f32>(0.0, 0.0, 0.0, 0.0);
    }
    let view = homogeneous.xyz / homogeneous.w;
    if (!finite3(view)) {
        return vec4<f32>(0.0, 0.0, 0.0, 0.0);
    }
    return vec4<f32>(view, 1.0);
}

fn rejected(motion: vec2<f32>, depth: f32) -> vec4<f32> {
    return vec4<f32>(motion.x, motion.y, depth, 0.0);
}

fn invalid() -> vec4<f32> {
    return vec4<f32>(0.0, 0.0, 0.0, 0.0);
}

fn previous_tap_rejects(
    previous_reactive: texture_2d<f32>,
    previous_model: texture_2d<f32>,
    stored_depth: texture_2d<f32>,
    tap: vec2<i32>,
    extent: vec2<f32>,
    expected_distance: f32,
    previous_view_from_clip: mat4x4<f32>,
) -> bool {
    if (mask_rejects(previous_reactive, tap) || mask_rejects(previous_model, tap)) {
        return true;
    }
    let depth = textureLoad(stored_depth, tap, 0).x;
    if (!finite(depth) || depth <= 0.0 || depth > 1.0) {
        return true;
    }
    let uv = (vec2<f32>(tap) + vec2<f32>(0.5, 0.5)) / extent;
    let view = reconstruct_view(uv, depth, previous_view_from_clip);
    if (view.w < 0.5) {
        return true;
    }
    let actual_distance = -view.z;
    if (!finite(actual_distance) || actual_distance <= 0.0) {
        return true;
    }
    let tolerance = max(0.02, 0.005 * expected_distance);
    return abs(actual_distance - expected_distance) > tolerance;
}

@vertex
fn vertex(@builtin(vertex_index) i: u32) -> @builtin(position) vec4<f32> {
    let x = f32((i << 1u) & 2u);
    let y = f32(i & 2u);
    return vec4<f32>(x * 2.0 - 1.0, y * 2.0 - 1.0, 0.0, 1.0);
}

@fragment
fn fragment(@builtin(position) p: vec4<f32>) -> @location(0) vec4<f32> {
    let extent = params.extent_and_flags.xy;
    if (params.extent_and_flags.z < 0.5 || extent.x <= 0.0 || extent.y <= 0.0) {
        return invalid();
    }
    if (!dimensions_match(current_depth, extent)
        || !dimensions_match(previous_depth, extent)
        || !dimensions_match(current_reactive_mask, extent)
        || !dimensions_match(previous_reactive_mask, extent)
        || !dimensions_match(current_model_mask, extent)
        || !dimensions_match(previous_model_mask, extent))
    {
        return invalid();
    }

    let uv = p.xy / extent;
    let depth = textureLoad(current_depth, vec2<i32>(p.xy), 0).x;
    // Background (0) has no previous surface; reverse depth is otherwise (0, 1].
    if (!finite(depth) || depth <= 0.0 || depth > 1.0) {
        return invalid();
    }

    let ndc = vec4<f32>(2.0 * uv.x - 1.0, 1.0 - 2.0 * uv.y, depth, 1.0);
    let current_h = params.current_view_from_clip * ndc;
    if (!finite4(current_h) || abs(current_h.w) < 1e-8) {
        return invalid();
    }
    let current_view = current_h.xyz / current_h.w;

    let previous_h = params.previous_clip_from_current_view * vec4<f32>(current_view, 1.0);
    if (!finite4(previous_h) || previous_h.w <= 1e-8) {
        return invalid();
    }
    let previous_ndc = previous_h.xyz / previous_h.w;
    if (!finite3(previous_ndc) || previous_ndc.z < 0.0 || previous_ndc.z > 1.0) {
        return invalid();
    }

    let previous_uv = vec2<f32>(
        0.5 * previous_ndc.x + 0.5,
        0.5 - 0.5 * previous_ndc.y,
    );
    if (!finite(previous_uv.x)
        || !finite(previous_uv.y)
        || previous_uv.x < 0.0
        || previous_uv.x >= 1.0
        || previous_uv.y < 0.0
        || previous_uv.y >= 1.0)
    {
        return invalid();
    }

    // previous_uv is the RAW previous raster coordinate. It is used for every previous depth and
    // mask lookup. The stored motion is corrected to Bevy's unjittered UV convention.
    let raw_motion = uv - previous_uv;
    let motion = raw_motion - params.raster_jitter_delta.xy;
    if (!finite(motion.x) || !finite(motion.y)) {
        return invalid();
    }

    // Current masks are evaluated at this pixel; either lane rejects the sample.
    let current_pixel = vec2<i32>(floor(p.xy));
    if (mask_rejects(current_reactive_mask, current_pixel)
        || mask_rejects(current_model_mask, current_pixel))
    {
        return rejected(motion, previous_ndc.z);
    }

    // The previous footprint is the clamped 2x2 bilinear support around previous_uv.
    let base = vec2<i32>(floor(previous_uv * extent - vec2<f32>(0.5, 0.5)));
    let max_coord = vec2<i32>(i32(extent.x) - 1, i32(extent.y) - 1);
    let offsets = array<vec2<i32>, 4>(
        vec2<i32>(0, 0),
        vec2<i32>(1, 0),
        vec2<i32>(0, 1),
        vec2<i32>(1, 1),
    );
    let expected_view = reconstruct_view(previous_uv, previous_ndc.z, params.previous_view_from_clip);
    if (expected_view.w < 0.5) {
        return rejected(motion, previous_ndc.z);
    }
    let expected_distance = -expected_view.z;
    if (!finite(expected_distance) || expected_distance <= 0.0) {
        return rejected(motion, previous_ndc.z);
    }
    for (var i = 0; i < 4; i = i + 1) {
        let tap = clamp(base + offsets[i], vec2<i32>(0, 0), max_coord);
        if (previous_tap_rejects(
            previous_reactive_mask,
            previous_model_mask,
            previous_depth,
            tap,
            extent,
            expected_distance,
            params.previous_view_from_clip,
        )) {
            return rejected(motion, previous_ndc.z);
        }
    }

    // xy = current minus previous in UV units, z = expected previous reverse depth,
    // w = all specified current/previous checks passed. This is still a candidate.
    return vec4<f32>(motion.x, motion.y, previous_ndc.z, 1.0);
}
