//! Test-only proof that the native TAA node can run before the existing transparent pass.

use std::sync::Mutex;

use bevy::core_pipeline::core_3d::graph::{Core3d, Node3d};
use bevy::ecs::query::QueryItem;
use bevy::prelude::*;
use bevy::render::render_graph::{
    NodeRunError, RenderGraph, RenderGraphExt, RenderLabel, ViewNode, ViewNodeRunner,
};
use bevy::render::render_resource::Texture;
use bevy::render::renderer::RenderContext;
use bevy::render::view::ViewTarget;
use bevy::render::RenderApp;

use super::{AfterReferenceTaa, BeforeReferenceTaa, CaptureStore};

#[derive(Resource)]
pub(super) struct EarlyQuad(pub(super) Entity);

#[derive(Clone)]
struct FinalObservation {
    view: Entity,
    frame: u64,
    texture: Texture,
}

#[derive(Resource, Default)]
struct FinalStore(Mutex<Option<FinalObservation>>);

impl FinalStore {
    fn set(&self, observation: FinalObservation) {
        *self.0.lock().unwrap() = Some(observation);
    }
    fn get(&self) -> Option<FinalObservation> {
        self.0.lock().unwrap().clone()
    }
}

#[derive(Debug, Hash, PartialEq, Eq, Clone, RenderLabel)]
struct EarlyFinalLabel;

#[derive(Default)]
struct EarlyFinalNode;

impl ViewNode for EarlyFinalNode {
    type ViewQuery = &'static ViewTarget;

    fn run<'w>(
        &self,
        graph: &mut bevy::render::render_graph::RenderGraphContext,
        _context: &mut RenderContext<'w>,
        target: QueryItem<'w, '_, Self::ViewQuery>,
        world: &'w World,
    ) -> Result<(), NodeRunError> {
        if let Some(after) = world.resource::<CaptureStore>().snapshot().1 {
            world.resource::<FinalStore>().set(FinalObservation {
                view: graph.view_entity(),
                frame: after.frame,
                texture: target.main_texture().clone(),
            });
        }
        Ok(())
    }
}

/// Configure the early-TAA graph and add the transparent quad used by the GPU proof.
pub(super) fn configure_early_taa(app: &mut App) {
    let mesh = app
        .world_mut()
        .resource_mut::<Assets<Mesh>>()
        .add(Rectangle::new(1.0, 1.0));
    let material = app
        .world_mut()
        .resource_mut::<Assets<StandardMaterial>>()
        .add(StandardMaterial {
            base_color: Color::srgba(1.0, 0.0, 0.0, 0.5),
            alpha_mode: AlphaMode::Blend,
            unlit: true,
            ..default()
        });
    let quad = app
        .world_mut()
        .spawn((
            Mesh3d(mesh),
            MeshMaterial3d(material),
            Transform::from_xyz(0.0, 0.0, 0.5),
        ))
        .id();
    app.insert_resource(EarlyQuad(quad));

    let Some(render_app) = app.get_sub_app_mut(RenderApp) else {
        return;
    };
    render_app
        .insert_resource(FinalStore::default())
        .add_render_graph_node::<ViewNodeRunner<EarlyFinalNode>>(Core3d, EarlyFinalLabel)
        .add_render_graph_edges(
            Core3d,
            (
                Node3d::MainTransparentPass,
                EarlyFinalLabel,
                Node3d::EndMainPass,
            ),
        );
    let mut graph = render_app.world_mut().resource_mut::<RenderGraph>();
    let core = graph.get_sub_graph_mut(Core3d).expect("Core3d subgraph");
    relocate_taa(core);
}

pub(super) fn relocate_taa(graph: &mut RenderGraph) {
    graph
        .remove_node_edge(Node3d::StartMainPassPostProcessing, Node3d::MotionBlur)
        .expect("baseline StartPost -> MotionBlur edge");
    graph
        .remove_node_edge(Node3d::Taa, Node3d::Bloom)
        .expect("baseline Taa -> Bloom edge");
    graph
        .try_add_node_edge(Node3d::MainOpaquePass, Node3d::MotionBlur)
        .expect("opaque -> motion blur");
    graph
        .try_add_node_edge(AfterReferenceTaa, Node3d::MainTransmissivePass)
        .expect("after-taa -> transmissive");
    assert!(
        graph
            .iter_node_outputs(Node3d::StartMainPassPostProcessing)
            .expect("start post node")
            .any(|(_, n)| n.label == Node3d::Bloom.intern()),
        "BloomPlugin must already provide StartMainPassPostProcessing -> Bloom"
    );
}

pub(super) fn final_observation(
    world: &World,
) -> Option<(Entity, u64, bevy::render::render_resource::TextureId)> {
    let f = world.resource::<FinalStore>().get()?;
    Some((f.view, f.frame, f.texture.id()))
}

pub(super) fn final_texture(world: &World) -> Option<Texture> {
    world.resource::<FinalStore>().get().map(|o| o.texture)
}

#[cfg(test)]
mod tests;
