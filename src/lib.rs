use bitvec::vec::BitVec;
use pyo3::prelude::*;

type RoomId = u8;
type Coord = i8;
type PartId = u8;
type DoorKind = u8;

#[derive(FromPyObject, Clone)]
#[pyo3(from_item_all)]
struct Room {
    map: Vec<Vec<u8>>,
    doors: Vec<Vec<Door>>,
    connections: Vec<(PartId, PartId)>,
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

pub struct Environment {
    actions: Vec<Action>,
    frontier: Vec<Frontier>,
    room_used: BitVec, // whether each room has been used
}

// Action: a placement of a room. The top-left corner is placed at (x, y) on the map.
pub struct Action {
    room: RoomId,
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
    fn new(rooms: &[Room], map_size: (Coord, Coord)) -> Self {
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

        Self {
            rooms: rooms.to_vec(),
            map_size,
            min_x_cand,
            max_x_cand,
            min_y_cand,
            max_y_cand,
        }
    }

    // Check if placing room1 at (x1, y1) and room2 at (x2, y2) would cause an intersection.
    // This includes overlapping tiles, blocked or mismatched doors, or out-of-bounds placement of room2.
    fn has_intersection(
        &self,
        room_id1: RoomId,
        x1: Coord,
        y1: Coord,
        room_id2: RoomId,
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

#[pyclass]
pub struct Engine {
    rooms: Vec<Room>,
}

#[pymethods]
impl Engine {
    #[new]
    fn new(rooms: Vec<Room>) -> Self {
        Self { rooms }
    }
}

#[pymodule]
fn map_gen(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Engine>()?;
    Ok(())
}
