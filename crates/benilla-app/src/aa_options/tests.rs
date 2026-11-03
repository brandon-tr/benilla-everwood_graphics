use super::*;

#[test]
fn selections_parse_and_map_msaa() {
    assert_eq!(AaSelection::parse("0"), AaSelection::Off);
    assert_eq!(AaSelection::parse("1"), AaSelection::Fallback);
    assert_eq!(AaSelection::parse("2"), AaSelection::Native);
    assert_eq!(AaSelection::parse("bogus"), AaSelection::Fallback);
    assert_eq!(
        camera_msaa_for_aa(AaSelection::Off, Msaa::Sample4, false),
        Msaa::Off
    );
    assert_eq!(
        camera_msaa_for_aa(AaSelection::Fallback, Msaa::Sample4, false),
        Msaa::Sample4
    );
    assert_eq!(
        camera_msaa_for_aa(AaSelection::Native, Msaa::Sample4, false),
        Msaa::Sample4
    );
    assert_eq!(
        camera_msaa_for_aa(AaSelection::Native, Msaa::Sample4, true),
        Msaa::Off
    );
}

/// The native path is single-sample only when the implementation is actually ready; otherwise the
/// configured fallback is untouched.
#[test]
fn native_selection_needs_readiness_for_single_sample() {
    use benilla_world::aa::AaReadiness;
    let ready = AaReadiness::with_native_taa(true);
    let not_ready = AaReadiness::with_native_taa(false);
    assert_eq!(
        boot_camera_msaa(AaSelection::Native, Msaa::Sample4, Some(&ready)),
        Msaa::Off
    );
    assert_eq!(
        boot_camera_msaa(AaSelection::Native, Msaa::Sample4, Some(&not_ready)),
        Msaa::Sample4
    );
    assert_eq!(
        boot_camera_msaa(AaSelection::Native, Msaa::Sample4, None),
        Msaa::Sample4
    );
    assert_eq!(
        boot_camera_msaa(AaSelection::Off, Msaa::Sample4, Some(&ready)),
        Msaa::Off
    );
    assert_eq!(
        boot_camera_msaa(AaSelection::Fallback, Msaa::Sample4, Some(&ready)),
        Msaa::Sample4
    );
}

#[test]
fn anti_aliasing_row_has_off_fallback_and_native() {
    let xml = include_str!("../../assets/ui/OptionsFrame.xml");
    let row = xml
        .split("<Frame name=\"$parentRowAntiAliasing\"")
        .nth(1)
        .expect("row");
    assert!(row.contains("\"gxAntiAliasing\""));
    assert!(row.contains("value = \"0\""));
    assert!(row.contains("value = \"1\""));
    assert!(row.contains("value = \"2\""));
    assert!(xml.contains("\"RowAntiAliasing\", \"RowRenderDistance\""));
}

#[test]
fn anti_aliasing_row_layout_is_stacked_and_restart_aware() {
    let xml = include_str!("../../assets/ui/OptionsFrame.xml");
    let aa = xml
        .split("<Frame name=\"$parentRowAntiAliasing\"")
        .nth(1)
        .expect("aa row");
    let distance = aa
        .split("<Frame name=\"$parentRowRenderDistance\"")
        .nth(1)
        .expect("distance follows");
    assert!(distance.contains("relativeTo=\"$parentRowAntiAliasing\""));
    assert!(distance.contains("\"farclip\""));
    assert!(!xml.contains(
        "</Frame>                                        <Frame name=\"$parentRowRenderDistance\""
    ));
    assert!(aa.contains("BENILLA_TOOLTIP_ANTI_ALIASING"));
    assert!(aa.contains("BENILLA_ADVGFX.aaNative"));
}
