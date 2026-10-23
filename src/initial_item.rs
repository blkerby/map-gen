use std::collections::HashMap;

use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use serde::Deserialize;

#[derive(Deserialize)]
struct Node {
    room_id: usize,
    node_id: usize,
    door: bool,
    item: bool,
    start: bool,
}

#[derive(Deserialize)]
struct Graph {
    edges: Vec<(usize, usize)>,
}

#[derive(Deserialize)]
struct Data {
    format: String,
    nodes: Vec<Node>,
    graphs: Vec<Graph>,
}

#[derive(Deserialize)]
struct Room {
    room_id: Option<usize>,
    doors: Vec<Vec<Door>>,
}

#[derive(Deserialize)]
struct Door {
    id: usize,
    direction: crate::common::Direction,
}

#[pyclass(module = "map_gen")]
pub struct InitialItemGraph {
    node_room: Vec<Option<usize>>,
    item: Vec<bool>,
    start: Vec<bool>,
    forward: Vec<Vec<Vec<usize>>>,
    reverse: Vec<Vec<Vec<usize>>>,
    door_node: Vec<Option<usize>>,
    opposite_offset: Vec<usize>,
    opposite_count: Vec<usize>,
    room_count: usize,
}

fn reachable(
    start: usize,
    edges: &[Vec<usize>],
    crossing: &[Option<usize>],
    node_room: &[Option<usize>],
    placed: &[bool],
) -> Vec<bool> {
    let mut seen = vec![false; edges.len()];
    let mut stack = vec![start];
    seen[start] = true;
    while let Some(node) = stack.pop() {
        for &next in edges[node].iter().chain(crossing[node].iter()) {
            if !seen[next] && node_room[next].is_some_and(|room| placed[room]) {
                seen[next] = true;
                stack.push(next);
            }
        }
    }
    seen
}

#[pymethods]
impl InitialItemGraph {
    #[new]
    fn new(rooms_json: &str, graph_json: &str) -> PyResult<Self> {
        let rooms: Vec<Room> = serde_json::from_str(rooms_json)
            .map_err(|error| PyValueError::new_err(error.to_string()))?;
        let data: Data = serde_json::from_str(graph_json)
            .map_err(|error| PyValueError::new_err(error.to_string()))?;
        if data.format != "initial-item-graph-v1" || data.graphs.is_empty() {
            return Err(PyValueError::new_err("unsupported initial-item graph data"));
        }
        let room_indices: HashMap<_, _> = rooms
            .iter()
            .enumerate()
            .filter_map(|(idx, room)| room.room_id.map(|id| (id, idx)))
            .collect();
        let mut nodes = HashMap::new();
        for (idx, node) in data.nodes.iter().enumerate() {
            if nodes.insert((node.room_id, node.node_id), idx).is_some() {
                return Err(PyValueError::new_err("duplicate initial-item node"));
            }
        }
        let count = data.nodes.len();
        let mut forward = Vec::new();
        let mut reverse = Vec::new();
        for graph in &data.graphs {
            let mut outgoing = vec![vec![]; count];
            let mut incoming = vec![vec![]; count];
            for &(from, to) in &graph.edges {
                if from >= count
                    || to >= count
                    || data.nodes[from].room_id != data.nodes[to].room_id
                {
                    return Err(PyValueError::new_err("invalid initial-item room edge"));
                }
                outgoing[from].push(to);
                incoming[to].push(from);
            }
            forward.push(outgoing);
            reverse.push(incoming);
        }
        let mut direction_nodes: [Vec<Option<usize>>; 4] = std::array::from_fn(|_| vec![]);
        for room in &rooms {
            for door in room.doors.iter().flatten() {
                let node = if let Some(room_id) = room.room_id {
                    let idx = *nodes.get(&(room_id, door.id)).ok_or_else(|| {
                        PyValueError::new_err(format!(
                            "missing initial-item door ({room_id}, {})",
                            door.id
                        ))
                    })?;
                    if !data.nodes[idx].door {
                        return Err(PyValueError::new_err(
                            "initial-item door node is not marked as a door",
                        ));
                    }
                    Some(idx)
                } else {
                    None
                };
                direction_nodes[door.direction as usize].push(node);
            }
        }
        let counts: Vec<_> = direction_nodes.iter().map(Vec::len).collect();
        let offsets = [
            0,
            counts[0],
            counts[0] + counts[1],
            counts[0] + counts[1] + counts[2],
        ];
        let mut opposite_offset = vec![];
        let mut opposite_count = vec![];
        for direction in 0..4 {
            opposite_offset.extend(std::iter::repeat_n(
                offsets[direction ^ 1],
                counts[direction],
            ));
            opposite_count.extend(std::iter::repeat_n(
                counts[direction ^ 1],
                counts[direction],
            ));
        }
        Ok(Self {
            node_room: data
                .nodes
                .iter()
                .map(|node| room_indices.get(&node.room_id).copied())
                .collect(),
            item: data.nodes.iter().map(|node| node.item).collect(),
            start: data.nodes.iter().map(|node| node.start).collect(),
            forward,
            reverse,
            door_node: direction_nodes.into_iter().flatten().collect(),
            opposite_offset,
            opposite_count,
            room_count: rooms.len(),
        })
    }

    fn evaluate<'py>(
        &self,
        py: Python<'py>,
        rooms: PyReadonlyArray2<'py, u8>,
        door_matches: PyReadonlyArray2<'py, i16>,
        starts: PyReadonlyArray1<'py, i64>,
        graphs: PyReadonlyArray1<'py, i64>,
    ) -> PyResult<Bound<'py, PyArray1<i8>>> {
        let rooms = rooms.as_array();
        let door_matches = door_matches.as_array();
        let starts = starts.as_array();
        let graphs = graphs.as_array();
        let batch = rooms.nrows();
        if door_matches.dim() != (batch, self.door_node.len())
            || starts.len() != batch
            || graphs.len() != batch
        {
            return Err(PyValueError::new_err(
                "initial-item evaluation input shapes disagree",
            ));
        }
        let mut outcomes = Vec::with_capacity(batch);
        for episode in 0..batch {
            let start = starts[episode];
            let graph = graphs[episode];
            if start < 0
                || start as usize > self.node_room.len()
                || graph < 0
                || graph as usize >= self.forward.len()
            {
                return Err(PyValueError::new_err(
                    "initial-item scenario index is out of range",
                ));
            }
            if start == 0 {
                outcomes.push(-1);
                continue;
            }
            let start = start as usize - 1;
            if !self.start[start] || self.node_room[start].is_none() {
                return Err(PyValueError::new_err(
                    "initial-item start does not belong to this room set",
                ));
            }
            let mut placed = vec![false; self.room_count];
            for &room in rooms.row(episode) {
                if (room as usize) < self.room_count {
                    placed[room as usize] = true;
                }
            }
            if !placed[self.node_room[start].unwrap()] {
                outcomes.push(1);
                continue;
            }
            let mut crossing = vec![None; self.node_room.len()];
            for (door, &matched) in door_matches.row(episode).iter().enumerate() {
                if matched < 0 || matched as usize >= self.opposite_count[door] {
                    continue;
                }
                let target = self.opposite_offset[door] + matched as usize;
                if let (Some(from), Some(to)) = (self.door_node[door], self.door_node[target]) {
                    crossing[from] = Some(to);
                }
            }
            let outgoing = reachable(
                start,
                &self.forward[graph as usize],
                &crossing,
                &self.node_room,
                &placed,
            );
            // Physical door matches are reciprocal in both traversals.
            let incoming = reachable(
                start,
                &self.reverse[graph as usize],
                &crossing,
                &self.node_room,
                &placed,
            );
            outcomes.push(i8::from(
                !self
                    .item
                    .iter()
                    .enumerate()
                    .any(|(node, &item)| item && outgoing[node] && incoming[node]),
            ));
        }
        Ok(outcomes.into_pyarray(py))
    }
}
