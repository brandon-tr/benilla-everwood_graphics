use super::*;
use crate::lighting::MonkeyFrame;

fn app() -> App {
    let mut app = App::new();
    app.init_resource::<MonkeyFrame>()
        .init_resource::<super::super::GameClock>()
        .init_resource::<WowLighting>()
        .init_resource::<WowLightData>()
        .init_resource::<GxRetainedWindFrame>()
        .insert_resource(FireLightGain(0.0))
        .insert_resource(SpellLightGain(0.0))
        .insert_resource(super::super::LavaLightGain(0.0))
        .add_systems(Update, pack_monkey_frame);
    app
}

#[test]
fn publishes_exact_wind_rows_changed_and_frozen() {
    let mut app = app();
    {
        let mut frame = app.world_mut().resource_mut::<MonkeyFrame>();
        frame.wind_dir = [0.6, 0.8];
        frame.wind_base_heading = 1.25;
        frame.wind_gust = 0.7;
        frame.wind_travel = 1234.5;
        frame.sway_strength = 0.9;
        frame.grass_strength = 1.0;
        frame.tree_strength = 1.0;
    }
    app.update();
    let packed = app.world().resource::<WowLightData>().0.monkey;
    let snapshot = *app.world().resource::<GxRetainedWindFrame>();
    assert_eq!(snapshot.rows[4], packed[4]);
    assert_eq!(snapshot.rows[5], packed[5]);
    assert_eq!(snapshot.rows, packed);
    assert_eq!(snapshot.rows[4][2], 1.25);
    assert_eq!(snapshot.rows[5][0], 1234.5);

    app.world_mut().resource_mut::<MonkeyFrame>().wind_gust = 0.2;
    app.update();
    let changed = *app.world().resource::<GxRetainedWindFrame>();
    assert_ne!(changed.rows[4][3], snapshot.rows[4][3]);
    assert_eq!(
        changed.rows[5],
        app.world().resource::<WowLightData>().0.monkey[5]
    );

    app.update();
    assert_eq!(
        *app.world().resource::<GxRetainedWindFrame>(),
        changed,
        "frozen wind keeps exact rows"
    );
}

#[test]
fn retains_all_packed_rows_bit_for_bit() {
    let mut rows = [[0.0f32; 4]; crate::lighting::MONKEY_FRAME_ROWS];
    for (i, row) in rows.iter_mut().enumerate() {
        for (j, value) in row.iter_mut().enumerate() {
            *value = (i * 4 + j) as f32 + 0.25;
        }
    }
    rows[7][1] = -0.0;
    rows[12][3] = f32::from_bits(0x8000_0001);
    let frame = GxRetainedWindFrame::from_rows(&rows);
    for (i, row) in rows.iter().enumerate() {
        for (j, expected) in row.iter().enumerate() {
            assert_eq!(
                frame.rows[i][j].to_bits(),
                expected.to_bits(),
                "row {i} component {j}"
            );
        }
    }
    assert_eq!(
        std::mem::size_of::<GxRetainedWindFrame>(),
        crate::lighting::MONKEY_FRAME_ROWS * 16
    );
}

#[test]
fn publishes_complete_packed_frame_and_tracks_non_wind_changes() {
    let mut app = app();
    {
        let mut frame = app.world_mut().resource_mut::<MonkeyFrame>();
        frame.wind_dir = [0.6, 0.8];
        frame.wind_base_heading = 0.7;
        frame.wind_gust = 0.5;
        frame.wind_travel = 150.0;
        frame.sway_strength = 1.0;
        frame.grass_strength = 1.0;
        frame.tree_strength = 1.0;
        frame.rain_rate = 1.25;
        frame.moon_light = 0.75;
        frame.bender_count = 2;
        frame.benders[0] = [1.0, 2.0, 3.0, 4.0];
        frame.benders[1] = [5.0, 6.0, 7.0, 8.0];
    }
    app.world_mut().insert_resource(FireLightGain(0.25));
    app.update();
    let packed = app.world().resource::<WowLightData>().0.monkey;
    let snapshot = *app.world().resource::<GxRetainedWindFrame>();
    assert_eq!(snapshot.rows, packed);
    assert_eq!(snapshot.rows[7][0], 2.5, "final fractional fire-gain flag");
    assert_eq!(snapshot.rows[8], [1.0, 2.0, 3.0, 4.0]);
    assert_eq!(snapshot.rows[9], [5.0, 6.0, 7.0, 8.0]);
    let wind = (snapshot.rows[4], snapshot.rows[5]);
    {
        let mut frame = app.world_mut().resource_mut::<MonkeyFrame>();
        frame.benders[0] = [9.0, 10.0, 11.0, 12.0];
        frame.moon_light = 0.1;
    }
    app.update();
    let changed = *app.world().resource::<GxRetainedWindFrame>();
    assert_ne!(changed.rows, snapshot.rows);
    assert_eq!((changed.rows[4], changed.rows[5]), wind);
    assert_eq!(
        snapshot.rows[8],
        [1.0, 2.0, 3.0, 4.0],
        "copy snapshot is immutable"
    );
    app.world_mut().resource_mut::<MonkeyFrame>().bender_count = 0;
    app.update();
    let cleared = *app.world().resource::<GxRetainedWindFrame>();
    for row in &cleared.rows[8..16] {
        assert_eq!(*row, [0.0; 4], "cleared benders leave no stale rows");
    }
    app.update();
    assert_eq!(*app.world().resource::<GxRetainedWindFrame>(), cleared);
}

#[test]
fn packing_without_retained_frame_preserves_light_output() {
    let mut app = app();
    app.world_mut().remove_resource::<GxRetainedWindFrame>();
    app.world_mut().resource_mut::<MonkeyFrame>().rain_rate = 0.75;
    app.update();
    let packed = app.world().resource::<WowLightData>().0.monkey;
    assert_eq!(packed[6][0], 0.75);
    assert!(app.world().get_resource::<GxRetainedWindFrame>().is_none());
}
