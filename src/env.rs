use serde_json::Value as JsonValue;

struct Room {
    mask: Vec<Vec<u8>>,
    doors: Vec<Door>,
}

struct Door {
    direction: Direction,
    x: usize,
    y: usize,
    subtype: String,
}

enum Direction {
    Left,
    Right,
    Up,
    Down,
}
