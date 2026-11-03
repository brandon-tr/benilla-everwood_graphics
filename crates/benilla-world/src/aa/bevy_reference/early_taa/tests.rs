use super::*;
use bevy::render::render_graph::{EmptyNode, InternedRenderLabel, RenderGraph};

fn add(graph: &mut RenderGraph, label: impl RenderLabel) {
    graph.add_node(label, EmptyNode);
}

fn baseline() -> RenderGraph {
    let mut g = RenderGraph::default();
    for l in [
        Node3d::MainOpaquePass,
        Node3d::MainTransmissivePass,
        Node3d::MainTransparentPass,
        Node3d::EndMainPass,
        Node3d::StartMainPassPostProcessing,
        Node3d::MotionBlur,
        Node3d::Taa,
        Node3d::Bloom,
        Node3d::Tonemapping,
    ] {
        add(&mut g, l);
    }
    add(&mut g, BeforeReferenceTaa);
    add(&mut g, AfterReferenceTaa);
    add(&mut g, EarlyFinalLabel);
    g.add_node_edge(Node3d::MainOpaquePass, Node3d::MainTransmissivePass);
    g.add_node_edge(Node3d::MainTransmissivePass, Node3d::MainTransparentPass);
    g.add_node_edge(Node3d::MainTransparentPass, Node3d::EndMainPass);
    g.add_node_edge(Node3d::EndMainPass, Node3d::StartMainPassPostProcessing);
    g.add_node_edge(Node3d::StartMainPassPostProcessing, Node3d::Tonemapping);
    g.add_node_edge(Node3d::StartMainPassPostProcessing, Node3d::Bloom);
    g.add_node_edge(Node3d::Bloom, Node3d::Tonemapping);
    g.add_node_edge(Node3d::StartMainPassPostProcessing, Node3d::MotionBlur);
    g.add_node_edge(Node3d::MotionBlur, Node3d::Taa);
    g.add_node_edge(Node3d::MotionBlur, BeforeReferenceTaa);
    g.add_node_edge(BeforeReferenceTaa, Node3d::Taa);
    g.add_node_edge(Node3d::Taa, AfterReferenceTaa);
    g.add_node_edge(Node3d::Taa, Node3d::Bloom);
    g.add_node_edge(AfterReferenceTaa, Node3d::Bloom);
    g
}

fn has(g: &RenderGraph, a: impl RenderLabel, b: impl RenderLabel) -> bool {
    g.iter_node_outputs(a)
        .unwrap()
        .any(|(_, n)| n.label == b.intern())
}

fn reaches(
    g: &RenderGraph,
    from: impl RenderLabel,
    to: impl RenderLabel,
    seen: &mut Vec<InternedRenderLabel>,
) -> bool {
    let target = to.intern();
    let mut stack = vec![from.intern()];
    while let Some(l) = stack.pop() {
        if seen.contains(&l) {
            continue;
        }
        seen.push(l);
        for (_, n) in g.iter_node_outputs(l).unwrap() {
            if n.label == target {
                return true;
            }
            stack.push(n.label);
        }
    }
    false
}

#[test]
fn early_taa_moves_taa_before_transparent_without_cycles() {
    let mut g = baseline();
    relocate_taa(&mut g);
    assert!(!has(
        &g,
        Node3d::StartMainPassPostProcessing,
        Node3d::MotionBlur
    ));
    assert!(!has(&g, Node3d::Taa, Node3d::Bloom));
    assert!(has(&g, Node3d::MainOpaquePass, Node3d::MotionBlur));
    assert!(has(&g, AfterReferenceTaa, Node3d::MainTransmissivePass));
    assert!(has(&g, Node3d::StartMainPassPostProcessing, Node3d::Bloom));
    assert!(has(&g, Node3d::Taa, AfterReferenceTaa));
    assert!(has(&g, AfterReferenceTaa, Node3d::MainTransmissivePass));
    assert!(has(
        &g,
        Node3d::MainTransmissivePass,
        Node3d::MainTransparentPass
    ));
    assert!(has(&g, Node3d::MainTransparentPass, Node3d::EndMainPass));
    assert!(has(
        &g,
        Node3d::EndMainPass,
        Node3d::StartMainPassPostProcessing
    ));
    assert!(has(&g, Node3d::StartMainPassPostProcessing, Node3d::Bloom));
    let mut seen = Vec::new();
    assert!(!reaches(
        &g,
        Node3d::StartMainPassPostProcessing,
        Node3d::Taa,
        &mut seen
    ));
    assert!(!has_cycle(&g));
}

fn has_cycle(g: &RenderGraph) -> bool {
    g.iter_nodes().any(|n| {
        let mut seen = Vec::new();
        reaches(g, n.label, n.label, &mut seen)
    })
}

#[test]
fn cycle_check_finds_cycle_beyond_acyclic_entry() {
    let mut g = RenderGraph::default();
    for l in [
        Node3d::MainOpaquePass,
        Node3d::MainTransmissivePass,
        Node3d::MainTransparentPass,
    ] {
        add(&mut g, l);
    }
    g.add_node_edge(Node3d::MainOpaquePass, Node3d::MainTransmissivePass);
    g.add_node_edge(Node3d::MainTransmissivePass, Node3d::MainTransparentPass);
    g.add_node_edge(Node3d::MainTransparentPass, Node3d::MainTransmissivePass);
    assert!(has_cycle(&g));
}

use crate::aa::bevy_reference::{wait_reference_stage, BevyReferenceFixture};

fn close(a: [f32; 4], b: [f32; 4], tol: f32) -> bool {
    (0..3).all(|i| (a[i] - b[i]).abs() <= tol)
}

struct Frame {
    before: crate::aa::bevy_reference::ObservedCapture,
    after: crate::aa::bevy_reference::ObservedCapture,
    final_tex: bevy::render::render_resource::Texture,
}

fn checked_frame(fixture: &BevyReferenceFixture, reset: Option<bool>) -> Result<Frame, String> {
    if !fixture.sampling_ready() {
        return Err("sampling not ready".into());
    }
    let (before, after) = fixture.captures();
    let (before, after) = (
        before.ok_or("no before capture")?,
        after.ok_or("no after capture")?,
    );
    if before.view != after.view {
        return Err("view mismatch".into());
    }
    if before.frame != after.frame {
        return Err("frame mismatch".into());
    }
    if !before.taa_present || !after.taa_present {
        return Err("native TAA absent".into());
    }
    if before.texture.id() == after.texture.id() {
        return Err("native TAA did not flip".into());
    }
    if let Some(reset) = reset {
        if before.reset != reset || after.reset != reset {
            return Err(format!(
                "reset state {:?}/{:?} != {reset}",
                before.reset, after.reset
            ));
        }
    }
    let (view, frame, id) =
        final_observation(fixture.render_world()).ok_or("no final observation")?;
    let texture = final_texture(fixture.render_world()).ok_or("no final texture")?;
    if view != after.view || frame != after.frame {
        return Err(("final observation frame/view mismatch").into());
    }
    if id != after.texture.id() || texture.id() != after.texture.id() {
        return Err("final texture is not the TAA destination".into());
    }
    Ok(Frame {
        before,
        after,
        final_tex: texture,
    })
}

fn require_frame(fixture: &BevyReferenceFixture, reset: bool, stage: &str) -> Frame {
    match checked_frame(fixture, Some(reset)) {
        Ok(f) => f,
        Err(err) => {
            let (before, after) = fixture.captures();
            panic!(
                "{stage}: {err}; sampling={} before={:?} after={:?} final={:?}",
                fixture.sampling_ready(),
                before.map(|c| (c.frame, c.view, c.reset, c.taa_present, c.texture.id())),
                after.map(|c| (c.frame, c.view, c.reset, c.taa_present, c.texture.id())),
                final_observation(fixture.render_world())
            );
        }
    }
}

fn check_pixels(fixture: &BevyReferenceFixture, frame: &Frame, final_rgb: [f32; 4], stage: &str) {
    let source = fixture.read_texture(&frame.before.texture);
    let final_pixels = fixture.read_texture(&frame.final_tex);
    let blue = LinearRgba::from(Color::srgb(0.2, 0.6, 0.9)).to_f32_array();
    for (x, y) in [(30u32, 32u32), (32, 32), (34, 32), (32, 34)] {
        let i = (y * 64 + x) as usize;
        let s = source[i];
        if !close(s, blue, 8.0e-3) {
            panic!("{stage}: frame={} view={:?} before={:?} after={:?} final={:?} reset={:?}/{:?} pixel=({x},{y}) source={s:?} expected={blue:?}", frame.before.frame, frame.before.view, frame.before.texture.id(), frame.after.texture.id(), frame.final_tex.id(), frame.before.reset, frame.after.reset);
        }
        let f = final_pixels[i];
        if !close(f, final_rgb, 8.0e-3) {
            panic!("{stage}: frame={} view={:?} before={:?} after={:?} final={:?} reset={:?}/{:?} pixel=({x},{y}) final={f:?} expected={final_rgb:?}", frame.before.frame, frame.before.view, frame.before.texture.id(), frame.after.texture.id(), frame.final_tex.id(), frame.before.reset, frame.after.reset);
        }
    }
    let i = (2 * 64 + 2) as usize;
    if !close(final_pixels[i], [0.0, 0.0, 0.0, 1.0], 8.0e-3) {
        panic!(
            "{stage}: frame={} background={:?}",
            frame.before.frame, final_pixels[i]
        );
    }
}

#[test]
#[ignore = "Requires a GPU; run explicitly on the host PC"]
fn gpu_native_taa_before_transparency_readback() {
    let mut fixture = BevyReferenceFixture::new_with_setup(configure_early_taa);
    let blue = LinearRgba::from(Color::srgb(0.2, 0.6, 0.9)).to_f32_array();
    let blend = [blue[0] * 0.5 + 0.5, blue[1] * 0.5, blue[2] * 0.5, 1.0];

    // A: bounded warmup under reset; actual TAA and scene pixels must be ready.
    let first = wait_reference_stage(&mut fixture, "early TAA warmup", |fixture| {
        let f = checked_frame(fixture, Some(true)).ok()?;
        let s = fixture.read_texture(&f.before.texture);
        let out = fixture.read_texture(&f.final_tex);
        let i = 32 * 64 + 32;
        if !close(s[i], blue, 8.0e-3) || !close(out[i], blend, 8.0e-3) {
            return None;
        }
        Some(f)
    });
    check_pixels(&fixture, &first, blend, "A");

    // B: bounded wait for non-reset execution, then four consecutive accepted frames.
    fixture.release_reset();
    let mut prev = wait_reference_stage(&mut fixture, "early TAA steady", |fixture| {
        checked_frame(fixture, Some(false)).ok()
    });
    check_pixels(&fixture, &prev, blend, "B0");
    for n in 1..4 {
        fixture.update();
        let f = require_frame(&fixture, false, "B");
        assert_eq!(f.before.frame, prev.after.frame + 1, "B{n} consecutive");
        check_pixels(&fixture, &f, blend, "B");
        prev = f;
    }

    let quad = fixture.app.world().resource::<EarlyQuad>().0;
    // C: hide, first applied frame and two more, all without reset.
    fixture
        .app
        .world_mut()
        .entity_mut(quad)
        .insert(Visibility::Hidden);
    let mut prev: Option<Frame> = None;
    for n in 0..3 {
        fixture.update();
        let f = require_frame(&fixture, false, "C");
        if let Some(p) = prev {
            assert_eq!(f.before.frame, p.after.frame + 1, "C{n} consecutive");
        }
        assert!(
            !fixture
                .app
                .world()
                .get::<ViewVisibility>(quad)
                .expect("view visibility")
                .get(),
            "C hidden state"
        );
        check_pixels(&fixture, &f, blue, "C");
        prev = Some(f);
    }

    // D: visible behind opaque, three frames, no reset.
    fixture
        .app
        .world_mut()
        .entity_mut(quad)
        .insert((Visibility::Visible, Transform::from_xyz(0.0, 0.0, -0.5)));
    let mut prev: Option<Frame> = None;
    for n in 0..3 {
        fixture.update();
        let f = require_frame(&fixture, false, "D");
        if let Some(p) = prev {
            assert_eq!(f.before.frame, p.after.frame + 1, "D{n} consecutive");
        }
        assert!(
            fixture
                .app
                .world()
                .get::<ViewVisibility>(quad)
                .expect("view visibility")
                .get(),
            "D visible state"
        );
        check_pixels(&fixture, &f, blue, "D");
        prev = Some(f);
    }

    // E: front again, three frames, no reset.
    fixture
        .app
        .world_mut()
        .entity_mut(quad)
        .insert((Visibility::Visible, Transform::from_xyz(0.0, 0.0, 0.5)));
    let mut prev: Option<Frame> = None;
    for n in 0..3 {
        fixture.update();
        let f = require_frame(&fixture, false, "E");
        if let Some(p) = prev {
            assert_eq!(f.before.frame, p.after.frame + 1, "E{n} consecutive");
        }
        check_pixels(&fixture, &f, blend, "E");
        prev = Some(f);
    }
    println!("EARLY-TAA A/B/C/D/E PASS");
}
