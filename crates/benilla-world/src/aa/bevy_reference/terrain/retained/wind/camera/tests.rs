use super::*;

#[test]
fn camera_fixture_discriminates_wrong_previous_camera() {
    camera_discrimination_check();
}

#[test]
#[ignore = "Requires a GPU; run explicitly on the host PC"]
fn gpu_retained_wind_camera_fade_readback() {
    gpu_camera_check();
}
