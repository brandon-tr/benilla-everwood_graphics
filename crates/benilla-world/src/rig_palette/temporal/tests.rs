use super::*;

#[test]
fn previous_rig_layout_matches_the_declared_total() {
    assert_eq!(SLOT_META_OFFSET, 288);
    assert_eq!(SLOT_META_BYTES, 32_768);
    assert_eq!(SLOT_BASES_OFFSET, 33_056);
    assert_eq!(SLOT_ORIGINS_OFFSET, 41_248);
    assert_eq!(PALETTE_OFFSET, 74_016);
    assert_eq!(PALETTE_BYTES, 6_291_456);
    assert_eq!(PREVIOUS_RIG_BYTES, 6_365_472);
    assert_eq!(PREVIOUS_RIG_BYTES % 16, 0);
}
