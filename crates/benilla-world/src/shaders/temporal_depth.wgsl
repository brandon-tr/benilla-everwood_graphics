// Per-view opaque-depth publication for future native TAA: a single-sample, full-target copy of
// the world camera's reverse-Z depth. No averaging, linearization, sampler or colour input.

@group(0) @binding(0) var depth: texture_depth_2d;

@vertex
fn vertex(@builtin(vertex_index) i: u32) -> @builtin(position) vec4<f32> {
    let x = f32((i << 1u) & 2u);
    let y = f32(i & 2u);
    return vec4<f32>(x * 2.0 - 1.0, y * 2.0 - 1.0, 0.0, 1.0);
}

@fragment
fn fragment(@builtin(position) p: vec4<f32>) -> @location(0) f32 {
    // The scene's actual raster, including its reverse-depth values; background stays 0.
    return textureLoad(depth, vec2<i32>(p.xy), 0);
}
