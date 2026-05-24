use anyhow::{Result, bail};
use bitvec::vec::BitVec;
use pyo3::prelude::*;
use rand::RngExt;

type RoomIdx = u8;
type Coord = i8;
type PartIdx = u8;
type DoorKind = u8;

#[derive(FromPyObject, Clone)]
#[pyo3(from_item_all)]
struct Room {
    room_id: i64,
    map: Vec<Vec<u8>>,
    doors: Vec<Vec<Door>>,
    connections: Vec<(PartIdx, PartIdx)>,
}

#[derive(FromPyObject, Clone, Debug)]
#[pyo3(from_item_all)]
struct Door {
    direction: Direction,
    x: Coord,
    y: Coord,
    kind: DoorKind,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
enum Direction {
    Left,
    Right,
    Up,
    Down,
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

impl<'py> FromPyObject<'py> for Direction {
    fn extract_bound(obj: &Bound<'py, PyAny>) -> PyResult<Self> {
        let s: &str = obj.extract()?;
        match s {
            "left" => Ok(Direction::Left),
            "right" => Ok(Direction::Right),
            "up" => Ok(Direction::Up),
            "down" => Ok(Direction::Down),
            _ => Err(pyo3::exceptions::PyValueError::new_err(
                "expected one of: 'left', 'right', 'up', 'down'",
            )),
        }
    }
}

// Action: a placement of a room. The top-left corner is placed at (x, y) on the map.
#[derive(Copy, Clone)]
pub struct Action {
    room_idx: RoomIdx,
    x: Coord,
    y: Coord,
}

// Frontier: location of an unconnected door on the map.
pub struct Frontier {
    direction: Direction,
    x: Coord,
    y: Coord,
    candidates: Vec<Action>, // possible actions to connect to this frontier
}

fn get_door_position(door: &Door, x: Coord, y: Coord) -> (Coord, Coord) {
    match door.direction {
        Direction::Left => (x + door.x - 1, y + door.y),
        Direction::Right => (x + door.x + 1, y + door.y),
        Direction::Up => (x + door.x, y + door.y - 1),
        Direction::Down => (x + door.x, y + door.y + 1),
    }
}

struct IntersectionChecker {
    rooms: Vec<Room>,
    map_size: (Coord, Coord),
    min_x_cand: Vec<Coord>, // For each room, the minimum x coordinate where room can be placed without going out of bounds
    max_x_cand: Vec<Coord>, // For each room, the maximum x coordinate where room can be placed without going out of bounds
    min_y_cand: Vec<Coord>, // For each room, the minimum y coordinate where room can be placed without going out of bounds
    max_y_cand: Vec<Coord>, // For each room, the maximum y coordinate where room can be placed without going out of bounds
}

impl IntersectionChecker {
    fn new(rooms: &[Room], map_size: (Coord, Coord)) -> Result<Self> {
        let mut min_x_cand = vec![];
        let mut max_x_cand = vec![];
        let mut min_y_cand = vec![];
        let mut max_y_cand = vec![];

        for room in rooms {
            let mut min_x = Coord::MAX;
            let mut max_x = Coord::MIN;
            let mut min_y = Coord::MAX;
            let mut max_y = Coord::MIN;
            let room_width = room.map[0].len() as Coord;
            let room_height = room.map.len() as Coord;
            for y in 0..room_height {
                for x in 0..room_width {
                    if room.map[y as usize][x as usize] != 0 {
                        min_x = min_x.min(x as Coord);
                        max_x = max_x.max(x as Coord);
                        min_y = min_y.min(y as Coord);
                        max_y = max_y.max(y as Coord);
                    }
                }
            }
            for door in room.doors.iter().flatten() {
                let (door_x, door_y) = get_door_position(door, 0, 0);
                min_x = min_x.min(door_x);
                max_x = max_x.max(door_x);
                min_y = min_y.min(door_y);
                max_y = max_y.max(door_y);
            }
            min_x_cand.push(-min_x);
            max_x_cand.push(map_size.0 - 1 - max_x);
            min_y_cand.push(-min_y);
            max_y_cand.push(map_size.1 - 1 - max_y);
        }

        for i in 0..rooms.len() {
            if min_x_cand[i] > max_x_cand[i] || min_y_cand[i] > max_y_cand[i] {
                bail!(
                    "Room id {} (index {}) cannot fit within the map boundaries",
                    rooms[i].room_id,
                    i
                );
            }
        }
        Ok(Self {
            rooms: rooms.to_vec(),
            map_size,
            min_x_cand,
            max_x_cand,
            min_y_cand,
            max_y_cand,
        })
    }

    // Check if placing room1 at (x1, y1) and room2 at (x2, y2) would cause an intersection.
    // This includes overlapping tiles, blocked or mismatched doors, or out-of-bounds placement of room2.
    fn has_intersection(
        &self,
        room_id1: RoomIdx,
        x1: Coord,
        y1: Coord,
        room_id2: RoomIdx,
        x2: Coord,
        y2: Coord,
    ) -> bool {
        if x2 < self.min_x_cand[room_id2 as usize]
            || x2 > self.max_x_cand[room_id2 as usize]
            || y2 < self.min_y_cand[room_id2 as usize]
            || y2 > self.max_y_cand[room_id2 as usize]
        {
            return true;
        }

        let room1 = &self.rooms[room_id1 as usize];
        let room2 = &self.rooms[room_id2 as usize];
        for (dy, row) in room1.map.iter().enumerate() {
            for (dx, &tile) in row.iter().enumerate() {
                if tile != 0 {
                    let other_x = x1 - x2 + dx as Coord;
                    let other_y = y1 - y2 + dy as Coord;
                    if other_y >= 0
                        && other_x >= 0
                        && other_y < room2.map.len() as Coord
                        && other_x < room2.map[0].len() as Coord
                        && room2.map[other_y as usize][other_x as usize] != 0
                    {
                        return true; // Intersection detected
                    }
                }
            }
        }

        'outer: for door1 in room1.doors.iter().flatten() {
            let (door_x1, door_y1) = get_door_position(door1, x1, y1);
            let other_x = door_x1 - x2;
            let other_y = door_y1 - y2;
            if other_y >= 0
                && other_x >= 0
                && other_y < room2.map.len() as Coord
                && other_x < room2.map[0].len() as Coord
                && room2.map[other_y as usize][other_x as usize] != 0
            {
                for door2 in room2.doors.iter().flatten() {
                    let (door_x2, door_y2) = get_door_position(door2, x2, y2);
                    if door_x1 == door_x2
                        && door_y1 == door_y2
                        && door1.direction == door2.direction.opposite()
                        && door1.kind == door2.kind
                    {
                        continue 'outer; // Doors match, check next door1
                    }
                }
                return true; // Mismatched door
            }
        }

        'outer: for door2 in room2.doors.iter().flatten() {
            let (door_x2, door_y2) = get_door_position(door2, x2, y2);
            let other_x = door_x2 - x1;
            let other_y = door_y2 - y1;
            if other_y >= 0
                && other_x >= 0
                && other_y < room1.map.len() as Coord
                && other_x < room1.map[0].len() as Coord
                && room1.map[other_y as usize][other_x as usize] != 0
            {
                for door1 in room1.doors.iter().flatten() {
                    let (door_x1, door_y1) = get_door_position(door1, x1, y1);
                    if door_x1 == door_x2
                        && door_y1 == door_y2
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

struct CommonData {
    rooms: Vec<Room>,
    intersection_checker: IntersectionChecker,
}

pub struct Environment {
    rng: rand::rngs::StdRng, // for random choice of initial room placement
    actions: Vec<Action>,    // history of room placements so far
    frontier: Vec<Frontier>, // info about each unconnected door on the map
    room_used: BitVec,       // whether each room has been used
}

impl Environment {
    fn new(rooms: &[Room], common: &CommonData) -> Self {
        let mut env = Self {
            rng: rand::make_rng(),
            actions: vec![],
            frontier: vec![],
            room_used: BitVec::repeat(false, rooms.len()),
        };
        let action = env.get_initial_action(common);
        env.step(action, common);
        env
    }

    fn get_initial_action(&mut self, common: &CommonData) -> Action {
        // Select a room and position uniformly at random.
        let room_idx = self.rng.random_range(0..common.rooms.len() as RoomIdx);
        let x = self.rng.random_range(
            common.intersection_checker.min_x_cand[room_idx as usize]
                ..=common.intersection_checker.max_x_cand[room_idx as usize],
        );
        let y = self.rng.random_range(
            common.intersection_checker.min_y_cand[room_idx as usize]
                ..=common.intersection_checker.max_y_cand[room_idx as usize],
        );
        Action { room_idx, x, y }
    }

    fn step(&mut self, action: Action, common: &CommonData) {
        self.actions.push(action.clone());
        self.room_used.set(action.room_idx as usize, true);
        for frontier in self.frontier.iter_mut() {
            frontier.candidates.retain(|cand| {
                !common.intersection_checker.has_intersection(
                    action.room_idx,
                    action.x,
                    action.y,
                    cand.room_idx,
                    cand.x,
                    cand.y,
                )
            });
        }
    }
}

#[pyclass]
pub struct Engine {
    common_data: CommonData, // pre-computed data that can be shared across environments
    environments: Vec<Environment>, // list of parallel environments for batch processing
}

#[pymethods]
impl Engine {
    #[new]
    fn new(rooms: Vec<Room>, map_size: (Coord, Coord), batch_size: usize) -> PyResult<Self> {
        let common_data = CommonData {
            rooms: rooms.clone(),
            intersection_checker: IntersectionChecker::new(&rooms, map_size)?,
        };
        let mut environments = Vec::with_capacity(batch_size);
        for _ in 0..batch_size {
            environments.push(Environment::new(&rooms, &common_data));
        }
        Ok(Self {
            common_data,
            environments,
        })
    }

    fn get_candidates(&self, start: usize, end: usize) -> Vec<()> {
        let mut candidates = vec![];
        // for (i, frontier) in self.frontier.iter().enumerate() {
        //     for room_id in start as RoomIdx..end as RoomIdx {
        //         if !self.room_used[room_id as usize] {
        //             for x in -10..=10 {
        //                 for y in -10..=10 {
        //                     if !IntersectionChecker::new(&self.rooms, (100, 100)).has_intersection(
        //                         frontier.candidates[0].room,
        //                         frontier.candidates[0].x,
        //                         frontier.candidates[0].y,
        //                         room_id,
        //                         x,
        //                         y,
        //                     ) {
        //                         candidates[i].push(Action {
        //                             room_idx: room_id,
        //                             x,
        //                             y,
        //                         });
        //                     }
        //                 }
        //             }
        //         }
        //     }
        // }
        candidates
    }
}

#[pymodule]
fn map_gen(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Engine>()?;
    Ok(())
}
