use indexmap::IndexMap;
use petgraph::algo::toposort;
use petgraph::graph::{DiGraph, NodeIndex};
use petgraph::visit::Dfs;

use crate::ir::model::{Layer, LayerKind, Model};

/// Result of graph construction: topologically sorted layer indices + diagnostics.
pub struct GraphInfo {
    /// Layer IDs in topological order.
    pub topo_order: Vec<String>,
    /// Warning messages (non-fatal).
    pub warnings: Vec<String>,
}

/// Build the DAG, do topo sort, cycle detection, and reachability checks.
pub fn build_graph(model: &Model) -> Result<GraphInfo, GraphError> {
    let mut graph = DiGraph::<&str, ()>::new();
    let mut node_map: IndexMap<&str, NodeIndex> = IndexMap::new();

    // Add nodes
    for layer in &model.layers {
        let idx = graph.add_node(layer.id.as_str());
        node_map.insert(layer.id.as_str(), idx);
    }

    // Add edges
    for edge in &model.edges {
        let src = node_map.get(edge.source.as_str()).ok_or_else(|| GraphError {
            code: "E001",
            message: format!("connection references unknown layer `{}`", edge.source),
        })?;
        let tgt = node_map.get(edge.target.as_str()).ok_or_else(|| GraphError {
            code: "E001",
            message: format!("connection references unknown layer `{}`", edge.target),
        })?;
        graph.add_edge(*src, *tgt, ());
    }

    // Cycle detection + topological sort
    let sorted = toposort(&graph, None).map_err(|cycle| {
        let layer_id = graph[cycle.node_id()];
        GraphError {
            code: "E006",
            message: format!("cycle detected involving layer `{layer_id}`"),
        }
    })?;

    let topo_order: Vec<String> = sorted
        .iter()
        .map(|idx| graph[*idx].to_string())
        .collect();

    // Check E001: non-Input layers must have at least one incoming edge
    let mut warnings = Vec::new();
    for layer in &model.layers {
        if matches!(layer.kind, LayerKind::Input { .. }) {
            continue;
        }
        let idx = node_map[layer.id.as_str()];
        let has_input = graph
            .neighbors_directed(idx, petgraph::Direction::Incoming)
            .next()
            .is_some();
        if !has_input {
            return Err(GraphError {
                code: "E001",
                message: format!("layer `{}` has no input connection", layer.id),
            });
        }
    }

    // W001: check reachability from any Input layer
    let input_indices: Vec<NodeIndex> = model
        .layers
        .iter()
        .filter(|l| matches!(l.kind, LayerKind::Input { .. }))
        .filter_map(|l| node_map.get(l.id.as_str()).copied())
        .collect();

    let mut reachable = vec![false; graph.node_count()];
    for &start in &input_indices {
        let mut dfs = Dfs::new(&graph, start);
        while let Some(visited) = dfs.next(&graph) {
            reachable[visited.index()] = true;
        }
    }

    for layer in &model.layers {
        let idx = node_map[layer.id.as_str()];
        if !reachable[idx.index()] {
            warnings.push(format!("W001: layer `{}` is unreachable", layer.id));
        }
    }

    Ok(GraphInfo {
        topo_order,
        warnings,
    })
}

pub fn get_input_layers<'a>(model: &'a Model, edge_target: &str) -> Vec<&'a Layer> {
    let source_ids: Vec<&str> = model
        .edges
        .iter()
        .filter(|e| e.target == edge_target)
        .map(|e| e.source.as_str())
        .collect();
    model
        .layers
        .iter()
        .filter(|l| source_ids.contains(&l.id.as_str()))
        .collect()
}

#[derive(Debug)]
pub struct GraphError {
    pub code: &'static str,
    pub message: String,
}
