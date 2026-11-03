use bevy::asset::io::AssetSourceId;
use bevy::prelude::*;

/// Reads each file on disk back through the embedded source a `ShaderRef` uses, so a shader
/// that is unregistered, or registered from a submodule, fails here instead of drawing nothing.
#[test]
fn every_engine_shader_answers_at_its_embedded_path() {
    let mut app = App::new();
    app.add_plugins(bevy::asset::AssetPlugin::default());
    app.add_plugins(super::plugin);
    let server = app.world().resource::<AssetServer>();
    let source = server.get_source(AssetSourceId::from("embedded")).unwrap();

    let dir = concat!(env!("CARGO_MANIFEST_DIR"), "/src/shaders");
    let mut found = 0;
    for entry in std::fs::read_dir(dir).unwrap() {
        let name = entry.unwrap().file_name().to_string_lossy().into_owned();
        if !name.ends_with(".wgsl") {
            continue;
        }
        found += 1;
        let path = format!("benilla_world/shaders/{name}");
        let read = bevy::tasks::block_on(async {
            source.reader().read(std::path::Path::new(&path)).await
        });
        assert!(
            read.is_ok(),
            "embedded://{path} does not resolve — {name} is on disk but not registered in \
                 `shaders::plugin`, or was registered from a file that is not directly under src/"
        );
    }
    // 6 since decision 1264 retired `skybox.wgsl` (a WMO skybox draws on the shared
    // model lane, whose forced-far-depth branch lives in `benilla_assets`'
    // `wow_model.wgsl`); 7 since 1429 added `static_gx.wgsl` (the B1 retained pass);
    // 8 since 2016 added `sky_vertex.wgsl` (the sky's shared far-depth vertex stage);
    // 9 since the torch-shadow Phase 1 added `torch_depth.wgsl` (the interior depth map).
    // MONKEY (sky): 10 with `sky_fx.wgsl`, the sky's Enhanced/High library.
    // 11 with `temporal_depth.wgsl`, the future native-TAA opaque-depth copy.
    // 12 with `temporal_motion.wgsl`, the camera-only reprojection candidate.
    // 13 with `temporal_reactive.wgsl`, the opaque snapshot/reactive mask lane.
    // 14 with `temporal_rejection.wgsl`, the per-view model rejection mask.
    // 15 with `temporal_resolve.wgsl`, the private reconstructed-color resolve/seed pass.
    assert_eq!(found, 15, "the engine's shader set changed size");
}
