use std::sync::Arc;

use bevy::camera::primitives::Aabb;

use super::super::render::GxItemDraw;
use super::*;

fn draw(foliage_wind: bool) -> GxItemDraw {
    GxItemDraw {
        index_range: 0..3,
        texture: None,
        cutout: true,
        two_sided: false,
        foliage_wind,
        vertex_range: 0..3,
        group: None,
        order: 0,
        sidn: [0; 3],
        slot: 0,
        ext_night: false,
        enclosed: false,
    }
}

fn region(draws: Vec<GxItemDraw>, killed: Vec<u64>) -> Arc<GxCellDraw> {
    Arc::new(GxCellDraw {
        mesh: Handle::default(),
        origin: Vec3::new(3.0, 4.0, 5.0),
        aabb: Aabb::from_min_max(Vec3::new(-1.0, -2.0, -3.0), Vec3::new(1.0, 2.0, 3.0)),
        draws,
        killed,
        killed_rev: 0,
        groups: Vec::new(),
        sets: Vec::new(),
    })
}

fn selection(drawn: Vec<bool>) -> GxSel {
    GxSel {
        drawn,
        fog: Vec::new(),
    }
}

fn collect(world: &GxWorld) -> Vec<Option<TemporalWindRegion>> {
    let mut visits = Vec::new();
    world.visit_temporal_wind(|region| {
        visits.push(region);
        true
    });
    visits
}

#[test]
fn visible_live_wind_cell_emits_one_bound() {
    let mut world = GxWorld::default();
    world
        .cells
        .insert((0, 0), region(vec![draw(true)], Vec::new()));
    world.visible.push(GxDoodadVis::Cell((0, 0)));
    let visits = collect(&world);
    assert_eq!(visits.len(), 1);
    let Some(region) = visits[0] else {
        panic!("expected a region");
    };
    assert_eq!(region.origin, Vec3::new(3.0, 4.0, 5.0));
    assert_eq!(region.bounds.center, Vec3::ZERO.into());
    assert_eq!(region.bounds.half_extents, Vec3::new(1.0, 2.0, 3.0).into());
}

#[test]
fn rigid_killed_empty_and_unselected_wind_emit_nothing() {
    let mut world = GxWorld::default();
    world
        .cells
        .insert((0, 0), region(vec![draw(false)], Vec::new()));
    world.visible.push(GxDoodadVis::Cell((0, 0)));
    assert!(collect(&world).is_empty());

    let mut world = GxWorld::default();
    world
        .cells
        .insert((0, 0), region(vec![draw(true)], vec![1]));
    world.visible.push(GxDoodadVis::Cell((0, 0)));
    assert!(collect(&world).is_empty());

    let mut world = GxWorld::default();
    let mut empty = draw(true);
    empty.index_range = 3..3;
    world.cells.insert((0, 0), region(vec![empty], Vec::new()));
    world.visible.push(GxDoodadVis::Cell((0, 0)));
    assert!(collect(&world).is_empty());

    let mut world = GxWorld::default();
    let mut item = draw(true);
    item.group = Some(0);
    world.props.insert(
        Entity::from_raw_u32(1).unwrap(),
        region(vec![item], Vec::new()),
    );
    world.visible.push(GxDoodadVis::Prop(
        Entity::from_raw_u32(1).unwrap(),
        selection(vec![false]),
    ));
    assert!(collect(&world).is_empty());
}

#[test]
fn prop_and_wmo_selection_match_renderer_missing_bits() {
    let entity = Entity::from_raw_u32(1).unwrap();
    let mut item = draw(true);
    item.group = Some(1);
    let mut world = GxWorld::default();
    world
        .props
        .insert(entity, region(vec![item.clone()], Vec::new()));
    world
        .visible
        .push(GxDoodadVis::Prop(entity, selection(vec![true, true])));
    assert_eq!(collect(&world).len(), 1);

    let mut world = GxWorld::default();
    world
        .wmos
        .insert(entity, region(vec![item.clone()], Vec::new()));
    world.visible_wmos.push((entity, selection(Vec::new())));
    assert!(collect(&world).is_empty());

    let mut world = GxWorld::default();
    world
        .wmos
        .insert(entity, region(vec![item.clone()], Vec::new()));
    world
        .visible_wmos
        .push((entity, selection(vec![true, true])));
    assert_eq!(collect(&world).len(), 1);
}

#[test]
fn missing_referenced_region_is_inconsistent() {
    let mut world = GxWorld::default();
    world.visible.push(GxDoodadVis::Cell((9, 9)));
    let visits = collect(&world);
    assert_eq!(visits.len(), 1);
    assert!(visits[0].is_none());

    let mut world = GxWorld::default();
    world
        .visible_wmos
        .push((Entity::from_raw_u32(2).unwrap(), selection(vec![true])));
    let visits = collect(&world);
    assert_eq!(visits.len(), 1);
    assert!(visits[0].is_none());
}

#[test]
#[allow(clippy::reversed_empty_ranges)] // deliberately malformed published range
fn malformed_or_unselected_region_data_fails_closed() {
    let mut world = GxWorld::default();
    let mut item = draw(true);
    item.index_range = 5..2;
    world.cells.insert((0, 0), region(vec![item], Vec::new()));
    world.visible.push(GxDoodadVis::Cell((0, 0)));
    let visits = collect(&world);
    assert_eq!(visits.len(), 1);
    assert!(visits[0].is_none());
}

#[test]
#[allow(clippy::reversed_empty_ranges)] // deliberately malformed published range
fn malformed_draw_after_wind_rejects_region() {
    let mut malformed = draw(true);
    malformed.index_range = 5..2;
    let mut world = GxWorld::default();
    world
        .cells
        .insert((0, 0), region(vec![draw(true), malformed], Vec::new()));
    world.visible.push(GxDoodadVis::Cell((0, 0)));
    let visits = collect(&world);
    assert_eq!(visits.len(), 1);
    assert!(visits[0].is_none());
}

#[test]
fn visitor_false_stops_promptly_and_nonvisible_regions_are_ignored() {
    let mut world = GxWorld::default();
    world
        .cells
        .insert((0, 0), region(vec![draw(true)], Vec::new()));
    world
        .cells
        .insert((1, 0), region(vec![draw(true)], Vec::new()));
    world.visible.push(GxDoodadVis::Cell((0, 0)));
    world.visible.push(GxDoodadVis::Cell((1, 0)));
    let mut visits = 0;
    world.visit_temporal_wind(|_| {
        visits += 1;
        false
    });
    assert_eq!(visits, 1);

    world.visible.clear();
    world.visible.push(GxDoodadVis::Cell((0, 0)));
    assert_eq!(collect(&world).len(), 1);
}
