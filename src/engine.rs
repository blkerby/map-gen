use crate::{Action, CommonData, Coord, Environment, Room, RoomIdx};
use numpy::{Element, IntoPyArray, PyArray2, PyArrayMethods, PyReadonlyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use scoped_pool::Pool;
use std::cmp::min;
use std::thread;

fn pyarray2_from_flat_vec<'py, T: Element>(
    py: Python<'py>,
    data: Vec<T>,
    rows: usize,
    cols: usize,
) -> PyResult<Bound<'py, PyArray2<T>>> {
    data.into_pyarray(py).reshape([rows, cols])
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

fn chunk_size(len: usize, num_chunks: usize) -> usize {
    if len == 0 {
        1
    } else {
        len.div_ceil(num_chunks.max(1))
    }
}

#[pyclass]
pub struct Engine {
    common_data: CommonData,
    environments: Vec<Environment>,
    pool: Pool,
    num_threads: usize,
}

impl Drop for Engine {
    fn drop(&mut self) {
        self.pool.shutdown();
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
        let num_threads = min(requested_threads, num_environments.max(1));
        let rooms: Vec<Room> = serde_json::from_str(rooms_json)
            .map_err(|err| PyValueError::new_err(format!("failed to parse rooms JSON: {err}")))?;
        let common_data = CommonData::new(rooms)?;
        let mut environments = Vec::with_capacity(num_environments);
        for env_idx in 0..num_environments {
            environments.push(Environment::new(
                &common_data,
                map_size,
                seed ^ env_idx as u64,
            ));
        }

        Ok(Self {
            common_data,
            environments,
            pool: Pool::new(num_threads),
            num_threads,
        })
    }

    fn clear(&mut self, py: Python<'_>) {
        let common_data = &self.common_data;
        let chunk_size = chunk_size(self.environments.len(), self.num_threads);
        py.allow_threads(|| {
            self.pool.scoped(|scope| {
                for environments in self.environments.chunks_mut(chunk_size) {
                    scope.execute(move || {
                        for env in environments {
                            env.clear(common_data);
                        }
                    });
                }
            });
        });
    }

    fn initial_step(&mut self, py: Python<'_>) {
        let common_data = &self.common_data;
        let chunk_size = chunk_size(self.environments.len(), self.num_threads);
        py.allow_threads(|| {
            self.pool.scoped(|scope| {
                for environments in self.environments.chunks_mut(chunk_size) {
                    scope.execute(move || {
                        for env in environments {
                            env.initial_step(common_data);
                        }
                    });
                }
            });
        });
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
        let chunk_size = chunk_size(self.environments.len(), self.num_threads);
        let mut rows: Vec<Option<(Vec<RoomIdx>, Vec<Coord>, Vec<Coord>)>> =
            Vec::with_capacity(self.environments.len());
        rows.resize_with(self.environments.len(), || None);

        py.allow_threads(|| {
            self.pool.scoped(|scope| {
                for (environments, rows) in self
                    .environments
                    .chunks(chunk_size)
                    .zip(rows.chunks_mut(chunk_size))
                {
                    scope.execute(move || {
                        for (env, row) in environments.iter().zip(rows.iter_mut()) {
                            *row = Some((
                                env.actions.iter().map(|action| action.room_idx).collect(),
                                env.actions.iter().map(|action| action.x).collect(),
                                env.actions.iter().map(|action| action.y).collect(),
                            ));
                        }
                    });
                }
            });
        });

        let mut room_idx = Vec::with_capacity(rows.len());
        let mut room_x = Vec::with_capacity(rows.len());
        let mut room_y = Vec::with_capacity(rows.len());
        for row in rows {
            let (idx_row, x_row, y_row) = row.expect("scoped worker filled action row");
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
        if end > self.environments.len() {
            return Err(PyValueError::new_err(format!(
                "action arrays with length {} starting at {} exceed num_environments {}",
                room_idx.len(),
                start,
                self.environments.len(),
            )));
        }

        let actions: Vec<_> = room_idx
            .iter()
            .zip(room_x.iter())
            .zip(room_y.iter())
            .map(|((&room_idx, &x), &y)| Action { room_idx, x, y })
            .collect();
        let common_data = &self.common_data;
        let chunk_size = chunk_size(actions.len(), self.num_threads);

        py.allow_threads(|| {
            self.pool.scoped(|scope| {
                for (environments, actions) in self.environments[start..end]
                    .chunks_mut(chunk_size)
                    .zip(actions.chunks(chunk_size))
                {
                    scope.execute(move || {
                        for (env, &action) in environments.iter_mut().zip(actions.iter()) {
                            env.step(action, common_data);
                        }
                    });
                }
            });
        });

        Ok(())
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
        if start > end || end > self.environments.len() {
            return Err(PyValueError::new_err(format!(
                "environment range [{}, {}) is invalid for num_environments {}",
                start,
                end,
                self.environments.len()
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
        let common_data = &self.common_data;
        let chunk_size = chunk_size(num_environments, self.num_threads);
        let output_chunk_len = chunk_size * max_candidates;

        py.allow_threads(|| {
            self.pool.scoped(|scope| {
                for (((environments, room_idx), room_x), room_y) in self.environments[start..end]
                    .chunks_mut(chunk_size)
                    .zip(room_idx.chunks_mut(output_chunk_len))
                    .zip(room_x.chunks_mut(output_chunk_len))
                    .zip(room_y.chunks_mut(output_chunk_len))
                {
                    scope.execute(move || {
                        for (env_idx, env) in environments.iter_mut().enumerate() {
                            let candidates = env.get_candidates(common_data, max_candidates);
                            let row_start = env_idx * max_candidates;
                            for (candidate_idx, candidate) in candidates.iter().enumerate() {
                                let idx = row_start + candidate_idx;
                                room_idx[idx] = candidate.room_idx;
                                room_x[idx] = candidate.x;
                                room_y[idx] = candidate.y;
                            }
                        }
                    });
                }
            });
        });

        Ok((
            pyarray2_from_flat_vec(py, room_idx, num_environments, max_candidates)?,
            pyarray2_from_flat_vec(py, room_x, num_environments, max_candidates)?,
            pyarray2_from_flat_vec(py, room_y, num_environments, max_candidates)?,
        ))
    }
}
