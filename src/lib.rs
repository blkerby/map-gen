use bitvec::vec::BitVec;
use pyo3::prelude::*;

type RoomId = u8;
type Coord = u8;

#[derive(FromPyObject, Clone)]
#[pyo3(from_item_all)]
struct Room {
    map: Vec<Vec<u8>>,
    doors: Vec<Door>,
}

#[derive(FromPyObject, Clone, Debug)]
#[pyo3(from_item_all)]
struct Door {
    direction: Direction,
    x: usize,
    y: usize,
    subtype: String,
}

#[derive(Copy, Clone, Debug)]
enum Direction {
    Left,
    Right,
    Up,
    Down,
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
