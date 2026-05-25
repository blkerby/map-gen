use anyhow::Result;
use bitvec::vec::BitVec;
use crossbeam_channel as channel;
use hashbrown::HashMap;
use numpy::{Element, IntoPyArray, PyArray2, PyArrayMethods, PyReadonlyArray1};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use rand::prelude::*;
use rand::{RngExt, SeedableRng};
use serde::Deserialize;
use std::cmp::{max, min};
use std::marker::PhantomData;
use std::sync::Arc;
use std::thread::{self, JoinHandle};

type RoomIdx = u8;
type GeometryIdx = u8;
type ConnectionVariantIdx = u8;
type Coord = i8;
type PartIdx = u8;
type DoorKind = u8;
type DirDoorIdx = u8; // index of a door among all doors with the given direction, across all rooms

const NUM_DIRS: usize = 4; // left, right, up, down

fn pyarray2_from_flat_vec<'py, T: Element>(
    py: Python<'py>,
    data: Vec<T>,
    rows: usize,
    cols: usize,
) -> PyResult<Bound<'py, PyArray2<T>>> {
    data.into_pyarray(py).reshape([rows, cols])
}

#[derive(Clone, Deserialize)]
struct Room {
    map: Vec<Vec<u8>>,
    doors: Vec<Vec<Door>>,
    connections: Vec<(PartIdx, PartIdx)>,
}

#[derive(Clone, Debug, Deserialize)]
struct Door {
    direction: Direction,
    x: Coord,
    y: Coord,
    kind: DoorKind,
}

#[derive(Copy, Clone, Debug, Deserialize, PartialEq, Eq, Hash)]
#[serde(rename_all = "lowercase")]
#[repr(u8)]
enum Direction {
    Left = 0,
    Right = 1,
    Up = 2,
    Down = 3,
}

impl Direction {
    fn opposite(&self) -> Self {
        match self {
            Direction::Left => Direction::Right,
            Direction::Right => Direction::Left,
            Direction::Up => Direction::Down,
            Direction::Down => Direction::Up,
        }
    }
}

// Action: a placement of a room. The top-left corner is placed at (x, y) on the map.
#[derive(Copy, Clone, Debug)]
pub struct Action {
    room_idx: RoomIdx,
    x: Coord,
    y: Coord,
}

// Frontier: location of an unconnected door on the map.
#[derive(Debug)]
pub struct Frontier {
    dir_door_idx: DirDoorIdx,
    candidates: Vec<GeometryAction>, // possible geometry placements to connect to this frontier
}

// Get the coordinates of the tile behind a door:
fn get_behind_door_position(direction: Direction, x: Coord, y: Coord) -> (Coord, Coord) {
    match direction {
        Direction::Left => (x - 1, y),
        Direction::Right => (x + 1, y),
        Direction::Up => (x, y - 1),
        Direction::Down => (x, y + 1),
    }
}

struct RoomDoorData {
    x: Coord,
    y: Coord,
    direction: Direction,
    dir_door_idx: DirDoorIdx,
}

struct RoomData {
    geometry_idx: GeometryIdx,
    connection_variant_idx: ConnectionVariantIdx,
    doors: Vec<RoomDoorData>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
struct GeometryDoorData {
    x: Coord,
    y: Coord,
    direction: Direction,
    kind: DoorKind,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct GeometryKey {
    map: Vec<Vec<u8>>,
    doors: Vec<GeometryDoorData>,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct ConnectionsKey {
    connections: Vec<(PartIdx, PartIdx)>,
}

struct GeometryData {
    map: Vec<Vec<u8>>,
    doors: Vec<GeometryDoorData>,
    // Minimum and maximum x and y coordinates at which the room can be placed without going out of bounds.
    min_x: Coord,
    max_x: Coord,
    min_y: Coord,
    max_y: Coord,
}

struct GeometryDirDoorData {
    geometry_idx: GeometryIdx,
    x: Coord,
    y: Coord,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
struct GeometryAction {
    geometry_idx: GeometryIdx,
    x: Coord,
    y: Coord,
}

struct CommonData {
    room: Vec<RoomData>,
    geometry: Vec<GeometryData>,
    geometry_rooms: Vec<Vec<RoomIdx>>,
    geometry_connection_variants: Vec<Vec<ConnectionVariantIdx>>,
    connection_variant_rooms: Vec<Vec<RoomIdx>>,
    // set of pairs of geometry placements that would cause an intersection
    intersection_idx: Vec<u32>, // maps a pair of geometry ids to the index of their intersection bits in the intersection_bitvec
    intersection_bitvec: BitVec,
    // for each direction, a list of all doors in that direction across all unique geometries
    geometry_dir_door: [Vec<GeometryDirDoorData>; NUM_DIRS],
    // for each direction, number of room doors in that direction across all rooms
    num_room_dir_doors: [usize; NUM_DIRS],
}

impl GeometryKey {
    fn from_room(room: &Room) -> Self {
        let map = room.map.clone();
        let mut doors: Vec<_> = room
            .doors
            .iter()
            .flatten()
            .map(|door| GeometryDoorData {
                x: door.x,
                y: door.y,
                direction: door.direction,
                kind: door.kind,
            })
            .collect();
        doors.sort_by_key(|door| (door.direction as u8, door.x, door.y, door.kind));
        Self { map, doors }
    }
}

impl ConnectionsKey {
    fn from_room(room: &Room) -> Self {
        let mut connections = room.connections.clone();
        connections.sort_unstable();
        Self { connections }
    }
}

impl GeometryData {
    fn new(key: &GeometryKey) -> Result<Self> {
        let mut min_x = Coord::MAX;
        let mut max_x = Coord::MIN;
        let mut min_y = Coord::MAX;
        let mut max_y = Coord::MIN;
        let room_width = key.map[0].len() as Coord;
        let room_height = key.map.len() as Coord;
        for y in 0..room_height {
            for x in 0..room_width {
                if key.map[y as usize][x as usize] != 0 {
                    min_x = min_x.min(x);
                    max_x = max_x.max(x);
                    min_y = min_y.min(y);
                    max_y = max_y.max(y);
                }
            }
        }
        for door in key.doors.iter() {
            let (door_x, door_y) = get_behind_door_position(door.direction, door.x, door.y);
            min_x = min_x.min(door_x);
            max_x = max_x.max(door_x);
            min_y = min_y.min(door_y);
            max_y = max_y.max(door_y);
        }
        Ok(Self {
            map: key.map.clone(),
            doors: key.doors.clone(),
            min_x,
            max_x,
            min_y,
            max_y,
        })
    }
}

impl CommonData {
    fn new(rooms: Vec<Room>) -> Result<Self> {
        let mut room_data = vec![];
        let mut geometry_data = vec![];
        let mut geometry_rooms = vec![];
        let mut geometry_connection_variants = vec![];
        let mut connection_variant_rooms = vec![];
        let mut geometry_by_key = HashMap::new();
        let mut connection_variant_by_key = HashMap::new();
        let mut geometry_dir_door: [Vec<GeometryDirDoorData>; NUM_DIRS] =
            std::array::from_fn(|_| vec![]);
        let mut num_room_dir_doors = [0; NUM_DIRS];

        for (room_idx, room) in rooms.iter().enumerate() {
            let mut door_data = vec![];
            for door in room.doors.iter().flatten() {
                let dir_idx = door.direction as usize;
                let dir_door_idx = num_room_dir_doors[dir_idx] as DirDoorIdx;
                num_room_dir_doors[dir_idx] += 1;
                door_data.push(RoomDoorData {
                    x: door.x,
                    y: door.y,
                    direction: door.direction,
                    dir_door_idx,
                });
            }

            let geometry_key = GeometryKey::from_room(room);
            let geometry_idx = if let Some(&geometry_idx) = geometry_by_key.get(&geometry_key) {
                geometry_idx
            } else {
                let geometry_idx = geometry_data.len() as GeometryIdx;
                let geometry = GeometryData::new(&geometry_key)?;
                for door in geometry.doors.iter() {
                    geometry_dir_door[door.direction as usize].push(GeometryDirDoorData {
                        geometry_idx,
                        x: door.x,
                        y: door.y,
                    });
                }
                geometry_data.push(geometry);
                geometry_rooms.push(vec![]);
                geometry_connection_variants.push(vec![]);
                geometry_by_key.insert(geometry_key, geometry_idx);
                geometry_idx
            };

            let connections_key = ConnectionsKey::from_room(room);
            let connection_variant_idx = if let Some(&connection_variant_idx) =
                connection_variant_by_key.get(&(geometry_idx, connections_key.clone()))
            {
                connection_variant_idx
            } else {
                let connection_variant_idx = connection_variant_rooms.len() as ConnectionVariantIdx;
                connection_variant_rooms.push(vec![]);
                geometry_connection_variants[geometry_idx as usize].push(connection_variant_idx);
                connection_variant_by_key
                    .insert((geometry_idx, connections_key), connection_variant_idx);
                connection_variant_idx
            };

            geometry_rooms[geometry_idx as usize].push(room_idx as RoomIdx);
            connection_variant_rooms[connection_variant_idx as usize].push(room_idx as RoomIdx);
            room_data.push(RoomData {
                geometry_idx,
                connection_variant_idx,
                doors: door_data,
            });
        }

        let mut common = Self {
            room: room_data,
            geometry: geometry_data,
            geometry_rooms,
            geometry_connection_variants,
            connection_variant_rooms,
            intersection_idx: vec![],
            intersection_bitvec: BitVec::new(),
            geometry_dir_door,
            num_room_dir_doors,
        };
        common.build_intersection_set();
        println!(
            "Finished building intersection set with {} bits across {} geometries",
            common.intersection_bitvec.len(),
            common.geometry.len()
        );
        Ok(common)
    }

    fn build_intersection_set(&mut self) {
        self.intersection_idx
            .resize(self.geometry.len() * self.geometry.len(), 0);
        for geometry_idx1 in 0..self.geometry.len() {
            let geometry1 = &self.geometry[geometry_idx1];
            for geometry_idx2 in geometry_idx1..self.geometry.len() {
                let geometry2 = &self.geometry[geometry_idx2];
                let x0 = -geometry2.max_x + geometry1.min_x;
                let x1 = geometry1.max_x - geometry2.min_x;
                let y0 = -geometry2.max_y + geometry1.min_y;
                let y1 = geometry1.max_y - geometry2.min_y;
                let bit_idx = self.intersection_bitvec.len();
                self.intersection_idx[geometry_idx1 * self.geometry.len() + geometry_idx2] =
                    bit_idx as u32;
                for y in y0..=y1 {
                    for x in x0..=x1 {
                        let b = self.slow_has_geometry_intersection(
                            geometry_idx1 as GeometryIdx,
                            0,
                            0,
                            geometry_idx2 as GeometryIdx,
                            x,
                            y,
                        );
                        self.intersection_bitvec.push(b);
                    }
                }
            }
        }
    }

    // Fast method using the pre-computed intersection_set:
    fn has_geometry_intersection(
        &self,
        mut geometry_id1: GeometryIdx,
        mut x1: Coord,
        mut y1: Coord,
        mut geometry_id2: GeometryIdx,
        mut x2: Coord,
        mut y2: Coord,
    ) -> bool {
        if geometry_id1 > geometry_id2 {
            std::mem::swap(&mut geometry_id1, &mut geometry_id2);
            std::mem::swap(&mut x1, &mut x2);
            std::mem::swap(&mut y1, &mut y2);
        }
        let geometry1 = &self.geometry[geometry_id1 as usize];
        let geometry2 = &self.geometry[geometry_id2 as usize];
        let x = x2 - x1;
        let y = y2 - y1;
        let x0 = -geometry2.max_x + geometry1.min_x;
        let x1 = geometry1.max_x - geometry2.min_x;
        let y0 = -geometry2.max_y + geometry1.min_y;
        let y1 = geometry1.max_y - geometry2.min_y;
        if x < x0 || x > x1 || y < y0 || y > y1 {
            // Bounding boxes do not intersect, so the geometries cannot intersect.
            return false;
        }
        let w = x1 - x0 + 1;
        let i = self.intersection_idx
            [geometry_id1 as usize * self.geometry.len() + geometry_id2 as usize];
        let bit_idx = i as usize + (y - y0) as usize * w as usize + (x - x0) as usize;
        self.intersection_bitvec[bit_idx]
    }

    // Check if placing geometry1 at (x1, y1) and geometry2 at (x2, y2) would cause an intersection.
    // This includes overlapping tiles or blocked or mismatched doors.
    // Slow method for computing the intersection_set, used during start-up.
    fn slow_has_geometry_intersection(
        &self,
        geometry_id1: GeometryIdx,
        x1: Coord,
        y1: Coord,
        geometry_id2: GeometryIdx,
        x2: Coord,
        y2: Coord,
    ) -> bool {
        let geometry1 = &self.geometry[geometry_id1 as usize];
        let geometry2 = &self.geometry[geometry_id2 as usize];
        for (dy, row) in geometry1.map.iter().enumerate() {
            for (dx, &tile) in row.iter().enumerate() {
                if tile != 0 {
                    let other_x = x1 - x2 + dx as Coord;
                    let other_y = y1 - y2 + dy as Coord;
                    if other_y >= 0
                        && other_x >= 0
                        && other_y < geometry2.map.len() as Coord
                        && other_x < geometry2.map[0].len() as Coord
                        && geometry2.map[other_y as usize][other_x as usize] != 0
                    {
                        return true; // Intersection detected
                    }
                }
            }
        }

        'outer: for door1 in geometry1.doors.iter() {
            let loc1 = DoorLocation::from_parts(door1.direction, door1.x, door1.y, x1, y1);
            let (door_x1, door_y1) =
                get_behind_door_position(door1.direction, x1 + door1.x, y1 + door1.y);
            let other_x = door_x1 - x2;
            let other_y = door_y1 - y2;
            if other_y >= 0
                && other_x >= 0
                && other_y < geometry2.map.len() as Coord
                && other_x < geometry2.map[0].len() as Coord
                && geometry2.map[other_y as usize][other_x as usize] != 0
            {
                for door2 in geometry2.doors.iter() {
                    let loc2 = DoorLocation::from_parts(door2.direction, door2.x, door2.y, x2, y2);
                    if loc1 == loc2
                        && door1.direction == door2.direction.opposite()
                        && door1.kind == door2.kind
                    {
                        continue 'outer; // Doors match, check next door1
                    }
                }
                return true; // Mismatched door
            }
        }

        'outer: for door2 in geometry2.doors.iter() {
            let loc2 = DoorLocation::from_parts(door2.direction, door2.x, door2.y, x2, y2);
            let (door_x2, door_y2) =
                get_behind_door_position(door2.direction, x2 + door2.x, y2 + door2.y);
            let other_x = door_x2 - x1;
            let other_y = door_y2 - y1;
            if other_y >= 0
                && other_x >= 0
                && other_y < geometry1.map.len() as Coord
                && other_x < geometry1.map[0].len() as Coord
                && geometry1.map[other_y as usize][other_x as usize] != 0
            {
                for door1 in geometry1.doors.iter() {
                    let loc1 = DoorLocation::from_parts(door1.direction, door1.x, door1.y, x1, y1);
                    if loc1 == loc2
                        && door1.direction == door2.direction.opposite()
                        && door1.kind == door2.kind
                    {
                        continue 'outer; // Doors match, check next door2
                    }
                }
                return true; // Mismatched door
            }
        }

        false // No intersection
    }
}

// DoorLocation: used as the key in the frontier hashmap to identify unconnected doors on the map.
// These are designed to match between the two sides of a door. A right-facing door gives the same
// DoorLocation as a left-facing door on the other side, and similarly for up/down doors.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
struct DoorLocation {
    x: Coord,
    y: Coord,
    vertical: bool,
}

impl DoorLocation {
    fn from_parts(
        direction: Direction,
        door_x: Coord,
        door_y: Coord,
        x0: Coord,
        y0: Coord,
    ) -> Self {
        let (x, y) = match direction {
            Direction::Left => (x0 + door_x, y0 + door_y),
            Direction::Right => (x0 + door_x + 1, y0 + door_y),
            Direction::Up => (x0 + door_x, y0 + door_y),
            Direction::Down => (x0 + door_x, y0 + door_y + 1),
        };
        let vertical = matches!(direction, Direction::Up | Direction::Down);
        Self { x, y, vertical }
    }

    // Get the DoorLocation for a door given the room placement, where (x0, y0) is the
    // location of the room's top-left corner on the map.
    fn new(door: &RoomDoorData, x0: Coord, y0: Coord) -> Self {
        Self::from_parts(door.direction, door.x, door.y, x0, y0)
    }
}

pub struct Environment {
    rng: rand::rngs::StdRng, // for randomly choosing the initial room placement
    map_size: (Coord, Coord),
    actions: Vec<Action>, // history of room placements so far
    frontier: HashMap<DoorLocation, Frontier>, // info about each unconnected door on the map
    // Grouped by door direction: for each door, the index of the matching door on the other side (or DirDoorIdx::MAX if none):
    door_matches: [Vec<DirDoorIdx>; NUM_DIRS],
    room_used: BitVec,                           // whether each room has been used
    geometry_unused_count: Vec<usize>, // number of unused room representatives for each geometry
    connection_variant_unused_count: Vec<usize>, // number of unused room representatives for each connection variant
}

impl Environment {
    fn new(common: &CommonData, map_size: (Coord, Coord), seed: u64) -> Self {
        Self {
            rng: rand::rngs::StdRng::seed_from_u64(seed),
            map_size,
            actions: vec![],
            frontier: HashMap::new(),
            door_matches: std::array::from_fn(|i| {
                vec![DirDoorIdx::MAX; common.num_room_dir_doors[i]]
            }),
            room_used: BitVec::repeat(false, common.room.len()),
            geometry_unused_count: common
                .geometry_rooms
                .iter()
                .map(|rooms| rooms.len())
                .collect(),
            connection_variant_unused_count: common
                .connection_variant_rooms
                .iter()
                .map(|rooms| rooms.len())
                .collect(),
        }
    }

    fn clear(&mut self, common: &CommonData) {
        self.actions.clear();
        self.frontier.clear();
        self.door_matches
            .iter_mut()
            .for_each(|matches| matches.fill(DirDoorIdx::MAX));
        self.room_used.fill(false);
        self.geometry_unused_count.clear();
        self.geometry_unused_count
            .extend(common.geometry_rooms.iter().map(|rooms| rooms.len()));
        self.connection_variant_unused_count.clear();
        self.connection_variant_unused_count.extend(
            common
                .connection_variant_rooms
                .iter()
                .map(|rooms| rooms.len()),
        );
    }

    fn initial_step(&mut self, common: &CommonData) {
        let action = self.get_initial_action(common);
        self.step(action, common);
    }

    fn get_initial_action(&mut self, common: &CommonData) -> Action {
        // Select a room and position uniformly at random.
        let room_idx = self.rng.random_range(0..common.room.len() as RoomIdx);
        let geometry_idx = common.room[room_idx as usize].geometry_idx;
        let geometry = &common.geometry[geometry_idx as usize];
        let min_x = -geometry.min_x;
        let max_x = self.map_size.0 - 1 - geometry.max_x;
        let min_y = -geometry.min_y;
        let max_y = self.map_size.1 - 1 - geometry.max_y;
        let x = self.rng.random_range(min_x..=max_x);
        let y = self.rng.random_range(min_y..=max_y);
        Action { room_idx, x, y }
    }

    fn choose_unused_room_in_connection_variant(
        &mut self,
        common: &CommonData,
        connection_variant_idx: ConnectionVariantIdx,
    ) -> Option<RoomIdx> {
        let remaining = self.connection_variant_unused_count[connection_variant_idx as usize];
        if remaining == 0 {
            return None;
        }
        let mut target = self.rng.random_range(0..remaining);
        for &room_idx in common.connection_variant_rooms[connection_variant_idx as usize].iter() {
            if self.room_used[room_idx as usize] {
                continue;
            }
            if target == 0 {
                return Some(room_idx);
            }
            target -= 1;
        }
        None
    }

    fn push_candidate_representatives(
        &mut self,
        common: &CommonData,
        candidate: GeometryAction,
        actions: &mut Vec<Action>,
    ) {
        for &connection_variant_idx in
            common.geometry_connection_variants[candidate.geometry_idx as usize].iter()
        {
            if self.connection_variant_unused_count[connection_variant_idx as usize] == 0 {
                continue;
            }
            if let Some(room_idx) =
                self.choose_unused_room_in_connection_variant(common, connection_variant_idx)
            {
                actions.push(Action {
                    room_idx,
                    x: candidate.x,
                    y: candidate.y,
                });
            }
        }
    }

    fn step(&mut self, action: Action, common: &CommonData) {
        self.actions.push(action);
        if action.room_idx >= common.room.len() as RoomIdx {
            // Dummy/invalid action: do nothing more.
            return;
        }
        let room = &common.room[action.room_idx as usize];
        let action_geometry_idx = room.geometry_idx;
        let connection_variant_idx = room.connection_variant_idx;
        assert!(!self.room_used[action.room_idx as usize]);
        self.room_used.set(action.room_idx as usize, true);
        self.geometry_unused_count[action_geometry_idx as usize] -= 1;
        self.connection_variant_unused_count[connection_variant_idx as usize] -= 1;

        // Remove the frontiers that the new room connects to (if any),
        // and update the frontier with the new unconnected doors of the new room.
        for door in room.doors.iter() {
            let door_loc = DoorLocation::new(door, action.x, action.y);
            if let Some(frontier) = self.frontier.remove(&door_loc) {
                // This frontier is now connected, so remove it and mark the doors as connected:
                let i1 = door.dir_door_idx;
                let i2 = frontier.dir_door_idx;
                self.door_matches[door.direction as usize][i1 as usize] = i2;
                self.door_matches[door.direction.opposite() as usize][i2 as usize] = i1;
            } else {
                // This door is not connected to any existing frontier, so it becomes a new frontier.
                // Check all doors with the given orientation, to list which ones could connect here.
                let (x1, y1) =
                    get_behind_door_position(door.direction, action.x + door.x, action.y + door.y);
                let mut candidates = vec![];
                'door: for opp_door in
                    common.geometry_dir_door[door.direction.opposite() as usize].iter()
                {
                    if self.geometry_unused_count[opp_door.geometry_idx as usize] == 0 {
                        // A geometry with no unused room representatives cannot be used again.
                        continue;
                    }
                    let room_x = x1 - opp_door.x;
                    let room_y = y1 - opp_door.y;
                    let geometry = &common.geometry[opp_door.geometry_idx as usize];
                    if room_x < -geometry.min_x
                        || room_x > self.map_size.0 - 1 - geometry.max_x
                        || room_y < -geometry.min_y
                        || room_y > self.map_size.1 - 1 - geometry.max_y
                    {
                        // The room cannot be placed at this position due to map boundaries.
                        continue;
                    }

                    for a in &self.actions {
                        let placed_geometry_idx = common.room[a.room_idx as usize].geometry_idx;
                        if common.has_geometry_intersection(
                            placed_geometry_idx,
                            a.x,
                            a.y,
                            opp_door.geometry_idx,
                            room_x,
                            room_y,
                        ) {
                            continue 'door;
                        }
                    }

                    // The geometry had no intersections with existing rooms, so it is a valid candidate at this frontier.
                    let candidate = GeometryAction {
                        geometry_idx: opp_door.geometry_idx,
                        x: room_x,
                        y: room_y,
                    };
                    candidates.push(candidate);
                }
                let frontier = Frontier {
                    dir_door_idx: door.dir_door_idx,
                    candidates,
                };
                self.frontier.insert(door_loc, frontier);
            }
        }

        // Filter existing frontiers to remove geometries blocked by the new room or with no unused representatives.
        let geometry_unused_count = &self.geometry_unused_count;
        for frontier in self.frontier.values_mut() {
            frontier.candidates.retain(|cand| {
                geometry_unused_count[cand.geometry_idx as usize] > 0
                    && !common.has_geometry_intersection(
                        action_geometry_idx,
                        action.x,
                        action.y,
                        cand.geometry_idx,
                        cand.x,
                        cand.y,
                    )
            });
        }
    }

    fn get_candidates(&mut self, common: &CommonData, max_candidates: usize) -> Vec<Action> {
        let smallest_frontier_size = self
            .frontier
            .values()
            .map(|frontier| frontier.candidates.len())
            .filter(|&x| x > 0)
            .min()
            .unwrap_or(1);
        let candidate_geometries = {
            let eligible_frontiers: Vec<&Frontier> = self
                .frontier
                .values()
                .filter(|frontier| frontier.candidates.len() == smallest_frontier_size)
                .collect();
            if eligible_frontiers.is_empty() {
                vec![]
            } else {
                let frontier = eligible_frontiers
                    .choose(&mut self.rng)
                    .expect("eligible_frontiers is not empty");
                frontier.candidates.clone()
            }
        };
        let mut candidates = Vec::with_capacity(candidate_geometries.len());
        for candidate in candidate_geometries {
            self.push_candidate_representatives(common, candidate, &mut candidates);
        }
        candidates.shuffle(&mut self.rng);
        candidates.truncate(max_candidates);
        candidates
    }
}

#[derive(Clone, Copy)]
struct OutputShard<T> {
    ptr: *mut T,
    len: usize,
    _marker: PhantomData<T>,
}

unsafe impl<T: Send> Send for OutputShard<T> {}

impl<T> OutputShard<T> {
    fn from_slice(slice: &mut [T]) -> Self {
        Self {
            ptr: slice.as_mut_ptr(),
            len: slice.len(),
            _marker: PhantomData,
        }
    }

    unsafe fn as_mut_slice<'a>(self) -> &'a mut [T] {
        unsafe { std::slice::from_raw_parts_mut(self.ptr, self.len) }
    }
}

type ActionRows = Vec<(Vec<RoomIdx>, Vec<Coord>, Vec<Coord>)>;

enum WorkerCommand {
    Clear,
    InitialStep,
    Step {
        local_start: usize,
        actions: Vec<Action>,
    },
    GetCandidates {
        local_start: usize,
        local_len: usize,
        max_candidates: usize,
        room_idx: OutputShard<RoomIdx>,
        room_x: OutputShard<Coord>,
        room_y: OutputShard<Coord>,
    },
    GetActions,
    Shutdown,
}

enum WorkerResponse {
    Done,
    Actions(ActionRows),
}

struct WorkerHandle {
    start: usize,
    len: usize,
    command_tx: channel::Sender<WorkerCommand>,
    response_rx: channel::Receiver<WorkerResponse>,
    join_handle: Option<JoinHandle<()>>,
}

impl WorkerHandle {
    fn end(&self) -> usize {
        self.start + self.len
    }

    fn send(&self, command: WorkerCommand) -> PyResult<()> {
        self.command_tx
            .send(command)
            .map_err(|_| PyRuntimeError::new_err("engine worker thread stopped unexpectedly"))
    }

    fn recv(&self) -> PyResult<WorkerResponse> {
        self.response_rx
            .recv()
            .map_err(|_| PyRuntimeError::new_err("engine worker thread stopped unexpectedly"))
    }

    fn recv_done(&self) -> PyResult<()> {
        match self.recv()? {
            WorkerResponse::Done => Ok(()),
            WorkerResponse::Actions(_) => Err(PyRuntimeError::new_err(
                "engine worker returned an unexpected response",
            )),
        }
    }

    fn shutdown(&mut self) {
        let _ = self.command_tx.send(WorkerCommand::Shutdown);
        if let Some(join_handle) = self.join_handle.take() {
            let _ = join_handle.join();
        }
    }
}

impl Drop for WorkerHandle {
    fn drop(&mut self) {
        self.shutdown();
    }
}

fn worker_loop(
    mut environments: Vec<Environment>,
    common_data: Arc<CommonData>,
    command_rx: channel::Receiver<WorkerCommand>,
    response_tx: channel::Sender<WorkerResponse>,
) {
    while let Ok(command) = command_rx.recv() {
        let response = match command {
            WorkerCommand::Clear => {
                for env in &mut environments {
                    env.clear(&common_data);
                }
                WorkerResponse::Done
            }
            WorkerCommand::InitialStep => {
                for env in &mut environments {
                    env.initial_step(&common_data);
                }
                WorkerResponse::Done
            }
            WorkerCommand::Step {
                local_start,
                actions,
            } => {
                let local_end = local_start + actions.len();
                debug_assert!(local_end <= environments.len());
                for (env, action) in environments[local_start..local_end].iter_mut().zip(actions) {
                    env.step(action, &common_data);
                }
                WorkerResponse::Done
            }
            WorkerCommand::GetCandidates {
                local_start,
                local_len,
                max_candidates,
                room_idx,
                room_x,
                room_y,
            } => {
                let local_end = local_start + local_len;
                debug_assert!(local_end <= environments.len());
                let room_idx = unsafe { room_idx.as_mut_slice() };
                let room_x = unsafe { room_x.as_mut_slice() };
                let room_y = unsafe { room_y.as_mut_slice() };
                debug_assert_eq!(room_idx.len(), local_len * max_candidates);
                debug_assert_eq!(room_x.len(), local_len * max_candidates);
                debug_assert_eq!(room_y.len(), local_len * max_candidates);

                for (env_idx, env) in environments[local_start..local_end].iter_mut().enumerate() {
                    let candidates = env.get_candidates(&common_data, max_candidates);
                    let row_start = env_idx * max_candidates;
                    for (candidate_idx, candidate) in candidates.iter().enumerate() {
                        let idx = row_start + candidate_idx;
                        room_idx[idx] = candidate.room_idx;
                        room_x[idx] = candidate.x;
                        room_y[idx] = candidate.y;
                    }
                }
                WorkerResponse::Done
            }
            WorkerCommand::GetActions => {
                let mut rows = Vec::with_capacity(environments.len());
                for env in &environments {
                    rows.push((
                        env.actions.iter().map(|action| action.room_idx).collect(),
                        env.actions.iter().map(|action| action.x).collect(),
                        env.actions.iter().map(|action| action.y).collect(),
                    ));
                }
                WorkerResponse::Actions(rows)
            }
            WorkerCommand::Shutdown => break,
        };

        if response_tx.send(response).is_err() {
            break;
        }
    }
}

fn spawn_worker(
    worker_idx: usize,
    start: usize,
    environments: Vec<Environment>,
    common_data: Arc<CommonData>,
) -> PyResult<WorkerHandle> {
    let len = environments.len();
    let (command_tx, command_rx) = channel::bounded(1);
    let (response_tx, response_rx) = channel::bounded(1);
    let join_handle = thread::Builder::new()
        .name(format!("map-gen-worker-{worker_idx}"))
        .spawn(move || worker_loop(environments, common_data, command_rx, response_tx))
        .map_err(|err| PyRuntimeError::new_err(format!("failed to spawn worker thread: {err}")))?;

    Ok(WorkerHandle {
        start,
        len,
        command_tx,
        response_rx,
        join_handle: Some(join_handle),
    })
}

fn requested_num_threads(num_threads: Option<usize>) -> PyResult<usize> {
    match num_threads {
        Some(0) => Err(PyValueError::new_err("num_threads must be greater than 0")),
        Some(num_threads) => Ok(num_threads),
        None => Ok(thread::available_parallelism()
            .map(|num_threads| num_threads.get())
            .unwrap_or(1)),
    }
}

fn checked_range_end(start: usize, len: usize) -> PyResult<usize> {
    start.checked_add(len).ok_or_else(|| {
        PyValueError::new_err(format!(
            "range start {start} with length {len} overflows usize"
        ))
    })
}

fn set_first_error(first_error: &mut Option<PyErr>, err: PyErr) {
    if first_error.is_none() {
        *first_error = Some(err);
    }
}

fn wait_for_done_responses(
    workers: &[WorkerHandle],
    sent_workers: Vec<usize>,
    mut first_error: Option<PyErr>,
) -> PyResult<()> {
    for worker_idx in sent_workers {
        if let Err(err) = workers[worker_idx].recv_done() {
            set_first_error(&mut first_error, err);
        }
    }

    if let Some(err) = first_error {
        Err(err)
    } else {
        Ok(())
    }
}

#[pyclass]
pub struct Engine {
    common_data: Arc<CommonData>, // pre-computed data that can be shared across environments
    workers: Vec<WorkerHandle>,   // fixed worker-owned environment shards
    num_environments: usize,
}

impl Drop for Engine {
    fn drop(&mut self) {
        for worker in &mut self.workers {
            worker.shutdown();
        }
    }
}

#[pymethods]
impl Engine {
    #[new]
    #[pyo3(signature = (rooms_json, map_size, num_environments, seed, num_threads=None))]
    fn new(
        rooms_json: &str,
        map_size: (Coord, Coord),
        num_environments: usize,
        seed: u64,
        num_threads: Option<usize>,
    ) -> PyResult<Self> {
        let requested_threads = requested_num_threads(num_threads)?;
        let worker_count = min(requested_threads, max(num_environments, 1));
        let rooms: Vec<Room> = serde_json::from_str(rooms_json)
            .map_err(|err| PyValueError::new_err(format!("failed to parse rooms JSON: {err}")))?;
        let common_data = Arc::new(CommonData::new(rooms)?);

        let base_shard_len = num_environments / worker_count;
        let remainder = num_environments % worker_count;
        let mut workers = Vec::with_capacity(worker_count);
        let mut start = 0;
        for worker_idx in 0..worker_count {
            let shard_len = base_shard_len + usize::from(worker_idx < remainder);
            let end = start + shard_len;
            let mut environments = Vec::with_capacity(shard_len);
            for env_idx in start..end {
                environments.push(Environment::new(
                    &common_data,
                    map_size,
                    seed ^ env_idx as u64,
                ));
            }
            workers.push(spawn_worker(
                worker_idx,
                start,
                environments,
                Arc::clone(&common_data),
            )?);
            start = end;
        }

        Ok(Self {
            common_data,
            workers,
            num_environments,
        })
    }

    fn clear(&mut self, py: Python<'_>) -> PyResult<()> {
        py.allow_threads(|| {
            let mut sent_workers = Vec::with_capacity(self.workers.len());
            let mut first_error = None;
            for (worker_idx, worker) in self.workers.iter().enumerate() {
                if let Err(err) = worker.send(WorkerCommand::Clear) {
                    set_first_error(&mut first_error, err);
                    break;
                }
                sent_workers.push(worker_idx);
            }

            wait_for_done_responses(&self.workers, sent_workers, first_error)
        })
    }

    fn initial_step(&mut self, py: Python<'_>) -> PyResult<()> {
        py.allow_threads(|| {
            let mut sent_workers = Vec::with_capacity(self.workers.len());
            let mut first_error = None;
            for (worker_idx, worker) in self.workers.iter().enumerate() {
                if let Err(err) = worker.send(WorkerCommand::InitialStep) {
                    set_first_error(&mut first_error, err);
                    break;
                }
                sent_workers.push(worker_idx);
            }

            wait_for_done_responses(&self.workers, sent_workers, first_error)
        })
    }

    #[allow(clippy::type_complexity)]
    fn get_actions<'py>(
        &self,
        py: Python<'py>,
    ) -> PyResult<(
        Bound<'py, PyArray2<RoomIdx>>,
        Bound<'py, PyArray2<Coord>>,
        Bound<'py, PyArray2<Coord>>,
    )> {
        let rows = py.allow_threads(|| {
            let mut sent_workers = Vec::with_capacity(self.workers.len());
            let mut first_error = None;
            for (worker_idx, worker) in self.workers.iter().enumerate() {
                if let Err(err) = worker.send(WorkerCommand::GetActions) {
                    set_first_error(&mut first_error, err);
                    break;
                }
                sent_workers.push(worker_idx);
            }

            let mut rows = Vec::with_capacity(self.num_environments);
            for worker_idx in sent_workers {
                match self.workers[worker_idx].recv() {
                    Ok(WorkerResponse::Actions(worker_rows)) => rows.extend(worker_rows),
                    Ok(WorkerResponse::Done) => {
                        set_first_error(
                            &mut first_error,
                            PyRuntimeError::new_err(
                                "engine worker returned an unexpected response",
                            ),
                        );
                    }
                    Err(err) => set_first_error(&mut first_error, err),
                }
            }

            if let Some(err) = first_error {
                Err(err)
            } else {
                Ok(rows)
            }
        })?;

        let mut room_idx = Vec::with_capacity(rows.len());
        let mut room_x = Vec::with_capacity(rows.len());
        let mut room_y = Vec::with_capacity(rows.len());
        for (idx_row, x_row, y_row) in rows {
            room_idx.push(idx_row);
            room_x.push(x_row);
            room_y.push(y_row);
        }
        Ok((
            PyArray2::from_vec2(py, &room_idx)
                .map_err(|_| PyValueError::new_err("environment action histories are ragged"))?,
            PyArray2::from_vec2(py, &room_x)
                .map_err(|_| PyValueError::new_err("environment action histories are ragged"))?,
            PyArray2::from_vec2(py, &room_y)
                .map_err(|_| PyValueError::new_err("environment action histories are ragged"))?,
        ))
    }

    fn step<'py>(
        &mut self,
        py: Python<'py>,
        room_idx: PyReadonlyArray1<'py, RoomIdx>,
        room_x: PyReadonlyArray1<'py, Coord>,
        room_y: PyReadonlyArray1<'py, Coord>,
        start: usize,
    ) -> PyResult<()> {
        let room_idx = room_idx
            .as_slice()
            .map_err(|_| PyValueError::new_err("room_idx must be a contiguous 1D numpy array"))?;
        let room_x = room_x
            .as_slice()
            .map_err(|_| PyValueError::new_err("room_x must be a contiguous 1D numpy array"))?;
        let room_y = room_y
            .as_slice()
            .map_err(|_| PyValueError::new_err("room_y must be a contiguous 1D numpy array"))?;

        if room_idx.len() != room_x.len() || room_idx.len() != room_y.len() {
            return Err(PyValueError::new_err(format!(
                "room_idx, room_x, and room_y must have the same length; got {}, {}, and {}",
                room_idx.len(),
                room_x.len(),
                room_y.len()
            )));
        }

        let end = checked_range_end(start, room_idx.len())?;
        if end > self.num_environments {
            return Err(PyValueError::new_err(format!(
                "action arrays with length {} starting at {} exceed num_environments {}",
                room_idx.len(),
                start,
                self.num_environments,
            )));
        }

        let actions: Vec<_> = room_idx
            .iter()
            .zip(room_x.iter())
            .zip(room_y.iter())
            .map(|((&room_idx, &x), &y)| Action { room_idx, x, y })
            .collect();

        py.allow_threads(|| {
            let mut sent_workers = Vec::with_capacity(self.workers.len());
            let mut first_error = None;
            for (worker_idx, worker) in self.workers.iter().enumerate() {
                let overlap_start = max(start, worker.start);
                let overlap_end = min(end, worker.end());
                if overlap_start >= overlap_end {
                    continue;
                }
                let action_start = overlap_start - start;
                let action_end = overlap_end - start;
                if let Err(err) = worker.send(WorkerCommand::Step {
                    local_start: overlap_start - worker.start,
                    actions: actions[action_start..action_end].to_vec(),
                }) {
                    set_first_error(&mut first_error, err);
                    break;
                }
                sent_workers.push(worker_idx);
            }

            wait_for_done_responses(&self.workers, sent_workers, first_error)
        })
    }

    #[allow(clippy::type_complexity)]
    fn get_candidates<'py>(
        &mut self,
        py: Python<'py>,
        max_candidates: usize,
        start: usize,
        end: usize,
    ) -> PyResult<(
        Bound<'py, PyArray2<RoomIdx>>,
        Bound<'py, PyArray2<Coord>>,
        Bound<'py, PyArray2<Coord>>,
    )> {
        if start > end || end > self.num_environments {
            return Err(PyValueError::new_err(format!(
                "environment range [{}, {}) is invalid for num_environments {}",
                start, end, self.num_environments
            )));
        }

        let num_environments = end - start;
        let output_len = num_environments
            .checked_mul(max_candidates)
            .ok_or_else(|| {
                PyValueError::new_err(format!(
                    "candidate output shape ({num_environments}, {max_candidates}) is too large"
                ))
            })?;
        let dummy_candidate = Action {
            room_idx: self.common_data.room.len() as RoomIdx, // an invalid room index to indicate no-op
            x: 0,
            y: 0,
        };

        let mut room_idx = vec![dummy_candidate.room_idx; output_len];
        let mut room_x = vec![dummy_candidate.x; output_len];
        let mut room_y = vec![dummy_candidate.y; output_len];

        py.allow_threads(|| {
            let mut sent_workers = Vec::with_capacity(self.workers.len());
            let mut first_error = None;
            for (worker_idx, worker) in self.workers.iter().enumerate() {
                let overlap_start = max(start, worker.start);
                let overlap_end = min(end, worker.end());
                if overlap_start >= overlap_end {
                    continue;
                }

                let output_row_start = overlap_start - start;
                let local_len = overlap_end - overlap_start;
                let output_start = output_row_start * max_candidates;
                let output_end = output_start + local_len * max_candidates;

                if let Err(err) = worker.send(WorkerCommand::GetCandidates {
                    local_start: overlap_start - worker.start,
                    local_len,
                    max_candidates,
                    room_idx: OutputShard::from_slice(&mut room_idx[output_start..output_end]),
                    room_x: OutputShard::from_slice(&mut room_x[output_start..output_end]),
                    room_y: OutputShard::from_slice(&mut room_y[output_start..output_end]),
                }) {
                    set_first_error(&mut first_error, err);
                    break;
                }
                sent_workers.push(worker_idx);
            }

            wait_for_done_responses(&self.workers, sent_workers, first_error)
        })?;

        Ok((
            pyarray2_from_flat_vec(py, room_idx, num_environments, max_candidates)?,
            pyarray2_from_flat_vec(py, room_x, num_environments, max_candidates)?,
            pyarray2_from_flat_vec(py, room_y, num_environments, max_candidates)?,
        ))
    }
}

#[pymodule]
fn map_gen(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Engine>()?;
    Ok(())
}
