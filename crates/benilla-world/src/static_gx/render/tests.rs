use super::*;

fn item(index_range: Range<u32>, cutout: bool) -> GxItemDraw {
    GxItemDraw {
        index_range,
        texture: None,
        cutout,
        two_sided: false,
        foliage_wind: false,
        vertex_range: 0..0,
        group: None,
        order: 0,
        sidn: [0; 3],
        slot: 0,
        ext_night: false,
        enclosed: false,
    }
}

#[test]
fn runs_split_between_rigid_and_wind() {
    let mut draws = vec![item(0..3, false), item(3..6, false), item(6..9, false)];
    draws[1].foliage_wind = true;
    let slots = vec![0, 0, 0];
    let runs = build_runs(&draws, &slots, &[]);
    assert_eq!(runs.len(), 3, "rigid/wind/rigid must not share a run");
    assert!(!runs[0].foliage_wind);
    assert!(runs[1].foliage_wind);
    assert!(!runs[2].foliage_wind);
    assert_eq!(runs[0].index_range, 0..3);
    assert_eq!(runs[1].index_range, 3..6);
    assert_eq!(runs[2].index_range, 6..9);
}

#[test]
fn runs_fuse_matching_wind_items() {
    let mut draws = vec![item(0..3, false), item(3..6, false), item(6..9, true)];
    draws[0].foliage_wind = true;
    draws[1].foliage_wind = true;
    draws[2].foliage_wind = true;
    let slots = vec![0, 0, 0];
    let runs = build_runs(&draws, &slots, &[]);
    assert_eq!(runs.len(), 2, "matching wind items fuse; cutout splits");
    assert!(runs[0].foliage_wind);
    assert_eq!(runs[0].index_range, 0..6);
    assert!(runs[1].foliage_wind);
    assert!(runs[1].cutout);
    assert_eq!(runs[1].index_range, 6..9);
}

#[test]
fn runs_fuse_live_items_and_split_at_kills() {
    let draws = vec![
        item(0..3, false),
        item(3..6, false),
        item(6..9, false),
        item(9..12, true), // bucket change
        item(12..15, true),
    ];
    let slots = vec![0, 0, 0, 0, 1]; // the last item binds another pool class
    let runs = build_runs(&draws, &slots, &[]);
    assert_eq!(runs.len(), 3, "opaque span fused; cutout split by slot");
    assert_eq!(runs[0].index_range, 0..9);
    assert_eq!(runs[1].index_range, 9..12);
    assert!(runs[1].cutout);
    assert_eq!(runs[2].slot, 1);
    // Kill the middle opaque item: the fused run splits around it.
    let runs = build_runs(&draws, &slots, &[0b010u64]);
    assert_eq!(runs.len(), 4);
    assert_eq!(runs[0].index_range, 0..3);
    assert_eq!(runs[1].index_range, 6..9);
    // Kill everything: nothing is submitted.
    assert!(build_runs(&draws, &slots, &[0b11111u64]).is_empty());
}
