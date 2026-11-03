// Opaque snapshot copy and reactive-mask comparison for the future native TAA lane.
// `0` means no detected color change in this interval, not rigid/velocity/history confidence.

@group(0) @binding(0) var opaque_color: texture_2d<f32>;
@group(0) @binding(1) var current_color: texture_2d<f32>;

fn finite(x: f32) -> bool {
    return x == x && abs(x) < 3.402823e38;
}

fn finite4(v: vec4<f32>) -> bool {
    return finite(v.x) && finite(v.y) && finite(v.z) && finite(v.w);
}

@vertex
fn vertex(@builtin(vertex_index) i: u32) -> @builtin(position) vec4<f32> {
    let x = f32((i << 1u) & 2u);
    let y = f32(i & 2u);
    return vec4<f32>(x * 2.0 - 1.0, y * 2.0 - 1.0, 0.0, 1.0);
}

@fragment
fn copy_fragment(@builtin(position) p: vec4<f32>) -> @location(0) vec4<f32> {
    return textureLoad(opaque_color, vec2<i32>(p.xy), 0);
}

@fragment
fn reactive_fragment(@builtin(position) p: vec4<f32>) -> @location(0) f32 {
    let dims = vec2<i32>(textureDimensions(opaque_color, 0));
    if (dims.x == 0 || dims.y == 0) {
        return 0.0;
    }
    let max_coord = dims - vec2<i32>(1, 1);
    let center = vec2<i32>(p.xy);
    var reactive = false;
    for (var dy: i32 = -1; dy <= 1; dy = dy + 1) {
        for (var dx: i32 = -1; dx <= 1; dx = dx + 1) {
            let coord = min(max(center + vec2<i32>(dx, dy), vec2<i32>(0, 0)), max_coord);
            let before = textureLoad(opaque_color, coord, 0);
            let after = textureLoad(current_color, coord, 0);
            if (!finite4(before) || !finite4(after)) {
                reactive = true;
            } else if (before.r != after.r
                || before.g != after.g
                || before.b != after.b
                || before.a != after.a)
            {
                reactive = true;
            }
        }
    }
    if (reactive) {
        return 1.0;
    }
    return 0.0;
}
