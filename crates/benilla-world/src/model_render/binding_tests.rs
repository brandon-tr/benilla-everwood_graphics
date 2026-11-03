use super::*;
use bevy::render::render_resource::BufferId;
use std::collections::HashSet;

fn key(light: BufferId, rig_taa: BufferId) -> MatKey {
    MatKey {
        light,
        rig_taa,
        texture: None,
        blend: ModelBlend::Opaque,
        two_sided: false,
        is_wmo: false,
        is_interior: false,
        is_emissive: false,
        is_additive: false,
        fade_variant: false,
        no_depth_write: false,
        no_depth_test: false,
        fog_policy: benilla_formats::FogPolicy::Scene,
        env_map: false,
        stage1: None,
        combine: None,
        shade: ShadeSel::Lit,
        batch_order: 0,
        uv_anim: None,
        rgb_anim: None,
        wmo_class: None,
        sidn: None,
        window: false,
        zfill: false,
        foliage_wind: false,
        sky_depth: false,
        instance: None,
    }
}

#[test]
fn cache_separates_previous_buffers_and_reuses_the_same_binding() {
    let light = BufferId::new();
    let previous_a = BufferId::new();
    let previous_b = BufferId::new();
    let mut keys = HashSet::new();
    assert!(keys.insert(key(light, previous_a)));
    assert!(!keys.insert(key(light, previous_a)));
    assert!(keys.insert(key(light, previous_b)));
    assert_eq!(keys.len(), 2);
}
