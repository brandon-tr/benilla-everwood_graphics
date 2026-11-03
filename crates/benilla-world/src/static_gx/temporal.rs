//! Temporal-wind bridge: publish the retained wind regions a view actually draws.
//!
//! The retained node draws its published selections, not `VisibleEntities`, so the temporal lane
//! consumes this CPU view of the same draw decisions. It intentionally over-covers: one coarse
//! baked region bound per selected region that contains at least one live wind draw.

use bevy::camera::primitives::Aabb;
use bevy::prelude::*;

use super::render::{kill_bit, GxCellDraw, GxDoodadVis, GxSel, GxWorld};

/// A retained region with at least one admitted wind draw, in the baked origin-relative form.
#[derive(Clone, Copy, Debug)]
pub(crate) struct TemporalWindRegion {
    pub(crate) origin: Vec3,
    pub(crate) bounds: Aabb,
}

enum RegionScan {
    Empty,
    Wind(TemporalWindRegion),
    Inconsistent,
}

fn scan_region(draw: &GxCellDraw, selection: Option<&GxSel>) -> RegionScan {
    let mut has_wind = false;
    for (index, item) in draw.draws.iter().enumerate() {
        if item.index_range.start > item.index_range.end {
            return RegionScan::Inconsistent;
        }
        if item.index_range.start == item.index_range.end {
            continue;
        }
        if kill_bit(&draw.killed, index) != 0 || !item.foliage_wind {
            continue;
        }
        if let (Some(selection), Some(group)) = (selection, item.group) {
            if !selection
                .drawn
                .get(usize::from(group))
                .copied()
                .unwrap_or(false)
            {
                continue;
            }
        }
        has_wind = true;
    }
    if has_wind {
        RegionScan::Wind(TemporalWindRegion {
            origin: draw.origin,
            bounds: draw.aabb,
        })
    } else {
        RegionScan::Empty
    }
}

impl GxWorld {
    /// Visit the selected, live retained regions containing at least one wind draw.
    ///
    /// `None` means a visible region's published data is missing or inconsistent. Returning
    /// `false` from `visit` stops the walk promptly. The visitor allocates no per-frame result
    /// vector and never inspects GPU residency.
    pub(crate) fn visit_temporal_wind(
        &self,
        mut visit: impl FnMut(Option<TemporalWindRegion>) -> bool,
    ) {
        for visible in &self.visible {
            let (draw, selection) = match visible {
                GxDoodadVis::Cell(cell) => match self.cells.get(cell) {
                    Some(draw) => (draw, None),
                    None => {
                        if !visit(None) {
                            return;
                        }
                        continue;
                    }
                },
                GxDoodadVis::Prop(entity, selection) => match self.props.get(entity) {
                    Some(draw) => (draw, Some(selection)),
                    None => {
                        if !visit(None) {
                            return;
                        }
                        continue;
                    }
                },
            };
            match scan_region(draw, selection) {
                RegionScan::Empty => {}
                RegionScan::Wind(region) => {
                    if !visit(Some(region)) {
                        return;
                    }
                }
                RegionScan::Inconsistent => {
                    if !visit(None) {
                        return;
                    }
                }
            }
        }

        for (entity, selection) in &self.visible_wmos {
            let Some(draw) = self.wmos.get(entity) else {
                if !visit(None) {
                    return;
                }
                continue;
            };
            match scan_region(draw, Some(selection)) {
                RegionScan::Empty => {}
                RegionScan::Wind(region) => {
                    if !visit(Some(region)) {
                        return;
                    }
                }
                RegionScan::Inconsistent => {
                    if !visit(None) {
                        return;
                    }
                }
            }
        }
    }
}

#[cfg(test)]
pub(crate) mod testkit {
    use super::super::render::GxItemDraw;
    use super::*;

    fn wind_draw() -> GxItemDraw {
        GxItemDraw {
            index_range: 0..3,
            texture: None,
            cutout: true,
            two_sided: false,
            foliage_wind: true,
            vertex_range: 0..3,
            group: None,
            order: 0,
            sidn: [0; 3],
            slot: 0,
            ext_night: false,
            enclosed: false,
        }
    }

    /// A published retained source with `count` distinct visible live wind cells around the
    /// origin. It is a test-only constructor; production has no setter.
    pub(crate) fn wind_source(count: usize) -> GxWorld {
        let mut world = GxWorld::default();
        for index in 0..count {
            let cell = (index as i32, 0);
            world.cells.insert(
                cell,
                std::sync::Arc::new(GxCellDraw {
                    mesh: Handle::default(),
                    origin: Vec3::new(0.0, 0.0, -8.0),
                    aabb: Aabb::from_min_max(Vec3::new(-1.0, -1.0, -1.0), Vec3::new(1.0, 1.0, 1.0)),
                    draws: vec![wind_draw()],
                    killed: Vec::new(),
                    killed_rev: 0,
                    groups: Vec::new(),
                    sets: Vec::new(),
                }),
            );
            world.visible.push(GxDoodadVis::Cell(cell));
        }
        world
    }
}

#[cfg(test)]
mod tests;
