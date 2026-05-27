use pyo3::prelude::*;

mod common;
mod engine;
mod environment;
mod scc_dag;

use engine::{Engine, EnvironmentGroup};

#[pymodule]
mod map_gen {
    #[pymodule_export]
    use super::{Engine, EnvironmentGroup};
}
