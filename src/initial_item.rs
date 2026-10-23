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
    room_nodes: Vec<Vec<usize>>,
    pub(crate) door_nodes: [Vec<Option<usize>>; 4],
    node_door: Vec<Option<(usize, usize)>>,
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

impl InitialItemGraph {
    pub(crate) fn from_rooms(rooms: &[crate::common::Room]) -> anyhow::Result<Self> {
        Self::build(
            rooms
                .iter()
                .map(|room| Room {
                    room_id: room.room_id,
                    doors: room
                        .doors
                        .iter()
                        .map(|doors| {
                            doors
                                .iter()
                                .map(|door| Door {
                                    id: door.id as usize,
                                    direction: door.direction,
                                })
                                .collect()
                        })
                        .collect(),
                })
                .collect(),
            include_str!("../data/initial-item-graph.json"),
        )
    }

    fn build(rooms: Vec<Room>, graph_json: &str) -> anyhow::Result<Self> {
        let data: Data = serde_json::from_str(graph_json)
            .map_err(|error| anyhow::Error::msg(error.to_string()))?;
        if data.format != "initial-item-graph-v1" || data.graphs.is_empty() {
            return Err(anyhow::Error::msg("unsupported initial-item graph data"));
        }
        let room_indices: HashMap<_, _> = rooms
            .iter()
            .enumerate()
            .filter_map(|(idx, room)| room.room_id.map(|id| (id, idx)))
            .collect();
        let mut nodes = HashMap::new();
        for (idx, node) in data.nodes.iter().enumerate() {
            if nodes.insert((node.room_id, node.node_id), idx).is_some() {
                return Err(anyhow::Error::msg("duplicate initial-item node"));
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
                    return Err(anyhow::Error::msg("invalid initial-item room edge"));
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
                        anyhow::Error::msg(format!(
                            "missing initial-item door ({room_id}, {})",
                            door.id
                        ))
                    })?;
                    if !data.nodes[idx].door {
                        return Err(anyhow::Error::msg(
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
        let mut room_nodes = vec![vec![]; rooms.len()];
        for (node, data_node) in data.nodes.iter().enumerate() {
            if let Some(&room) = room_indices.get(&data_node.room_id) {
                room_nodes[room].push(node);
            }
        }
        let mut node_door = vec![None; count];
        for (direction, door_nodes) in direction_nodes.iter().enumerate() {
            for (door, node) in door_nodes.iter().enumerate() {
                if let Some(node) = node {
                    node_door[*node] = Some((direction, door));
                }
            }
        }
        Ok(Self {
            room_nodes,
            node_door,
            door_nodes: direction_nodes.clone(),
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

    pub(crate) fn start_room(&self, start: usize) -> Option<usize> {
        self.node_room[start - 1]
    }

    pub(crate) fn validate_scenario(&self, start: i64, graph: i64) -> bool {
        start >= 0
            && (start as usize) <= self.node_room.len()
            && graph >= 0
            && (graph as usize) < self.forward.len()
            && (start == 0
                || (self.start[start as usize - 1] && self.node_room[start as usize - 1].is_some()))
    }
}

#[pymethods]
impl InitialItemGraph {
    #[new]
    fn new(rooms_json: &str, graph_json: &str) -> PyResult<Self> {
        let rooms: Vec<Room> = serde_json::from_str(rooms_json)
            .map_err(|error| PyValueError::new_err(error.to_string()))?;
        Ok(Self::build(rooms, graph_json)?)
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
                outcomes.push(0);
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

/// Labels are 0 (unreachable), 1 (reachable), or 2 (an item-visiting route exists).
#[derive(Clone, Debug)]
pub(crate) struct InitialItemState {
    pub start: usize,
    graph: usize,
    pub forward: Vec<u8>,
    pub reverse: Vec<u8>,
}

impl InitialItemState {
    pub fn new(start: usize, graph: usize, data: &InitialItemGraph) -> Self {
        Self {
            start,
            graph,
            forward: vec![0; data.node_room.len()],
            reverse: vec![0; data.node_room.len()],
        }
    }

    pub fn add_room(
        &mut self,
        room: usize,
        data: &InitialItemGraph,
        matches: &[Vec<crate::common::DirDoorIdx>; 4],
    ) {
        if self.start == 0 {
            return;
        }
        let start = self.start - 1;
        for (labels, edges) in [
            (&mut self.forward, &data.forward[self.graph]),
            (&mut self.reverse, &data.reverse[self.graph]),
        ] {
            let mut queue = Vec::new();
            if data.node_room[start] == Some(room) {
                labels[start] = if data.item[start] { 2 } else { 1 };
                queue.push(start);
            }
            for &node in &data.room_nodes[room] {
                if let Some((direction, door)) = data.node_door[node] {
                    let matched = matches[direction][door] as usize;
                    if let Some(Some(other)) = data.door_nodes[direction ^ 1].get(matched) {
                        if labels[*other] != 0 {
                            queue.push(*other);
                        }
                    }
                }
            }
            while let Some(node) = queue.pop() {
                let crossing = data.node_door[node].and_then(|(direction, door)| {
                    let matched = matches[direction][door] as usize;
                    data.door_nodes[direction ^ 1]
                        .get(matched)
                        .copied()
                        .flatten()
                });
                for next in edges[node].iter().copied().chain(crossing) {
                    let value = if data.item[next] { 2 } else { labels[node] };
                    if value > labels[next] {
                        labels[next] = value;
                        queue.push(next);
                    }
                }
            }
        }
    }

    pub fn has_item(&self, data: &InitialItemGraph) -> bool {
        self.start == 0
            || data
                .item
                .iter()
                .enumerate()
                .any(|(node, &item)| item && self.forward[node] > 0 && self.reverse[node] > 0)
    }

    pub fn frontier_label(&self, data: &InitialItemGraph, direction: usize, door: usize) -> u8 {
        data.door_nodes[direction][door]
            .map_or(0, |node| self.forward[node] * 3 + self.reverse[node])
    }
}
