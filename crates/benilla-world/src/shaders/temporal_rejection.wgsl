// Per-view model rejection mask: every covered pixel must reject camera-only temporal reuse.
// Fragment 1 is a conservative rejection, not a classification or geometry ID.

@vertex
fn vertex(@builtin(vertex_index) i: u32) -> @builtin(position) vec4<f32> {
    let x = f32((i << 1u) & 2u);
    let y = f32(i & 2u);
    return vec4<f32>(x * 2.0 - 1.0, y * 2.0 - 1.0, 0.0, 1.0);
}

@fragment
fn fragment() -> @location(0) f32 {
    return 1.0;
}
