use crate::dataloader::PyTarFileEntry;
use pyo3::exceptions::{PyIndexError, PyOSError, PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyIterator;
use rand::{rngs::StdRng, seq::SliceRandom, SeedableRng};
use std::collections::VecDeque;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::{Arc, Condvar, Mutex};
use std::thread::{self, JoinHandle};

#[pyclass(name = "_ShardOrder", frozen)]
pub(crate) struct ShardOrder(Vec<usize>);

impl ShardOrder {
    pub(crate) fn shuffled(count: usize, seed: u64) -> Self {
        let mut indices: Vec<_> = (0..count).collect();
        indices.shuffle(&mut StdRng::seed_from_u64(seed));
        Self(indices)
    }

    pub(crate) fn into_indices(self) -> std::vec::IntoIter<usize> {
        self.0.into_iter()
    }
}

#[pymethods]
impl ShardOrder {
    fn __len__(&self) -> usize {
        self.0.len()
    }

    fn __getitem__(&self, position: usize) -> PyResult<usize> {
        self.0
            .get(position)
            .copied()
            .ok_or_else(|| PyIndexError::new_err("Shard order position is out of bounds"))
    }
}

#[pyfunction]
pub(crate) fn _shuffled_shard_order(py: Python<'_>, count: usize, seed: u64) -> ShardOrder {
    py.detach(|| ShardOrder::shuffled(count, seed))
}

pub(crate) enum Sample {
    Tar(PyTarFileEntry),
    Parquet(Py<PyAny>),
}

#[derive(Default)]
struct State {
    ready: VecDeque<(Sample, usize)>,
    bytes: usize,
    finished: bool,
    closed: bool,
    error: Option<PyErr>,
}

struct Queue {
    state: Mutex<State>,
    changed: Condvar,
    max_bytes: usize,
    max_items: usize,
}

impl Queue {
    fn fill(
        &self,
        mut produce: impl FnMut(usize, usize) -> PyResult<Option<(Sample, usize)>>,
    ) -> PyResult<()> {
        loop {
            let state = self.state.lock().unwrap();
            let state = self
                .changed
                .wait_while(state, |state| {
                    !state.closed
                        && (state.bytes >= self.max_bytes || state.ready.len() >= self.max_items)
                })
                .unwrap();
            if state.closed {
                return Ok(());
            }
            let remaining_bytes = self.max_bytes - state.bytes;
            let remaining_items = self.max_items - state.ready.len();
            drop(state);

            let Some((sample, bytes)) = produce(remaining_bytes, remaining_items)? else {
                return Ok(());
            };
            let state = self.state.lock().unwrap();
            let mut state = self
                .changed
                .wait_while(state, |state| {
                    !state.closed
                        && !state.ready.is_empty()
                        && bytes > self.max_bytes.saturating_sub(state.bytes)
                })
                .unwrap();
            if state.closed {
                return Ok(());
            }
            state.bytes += bytes;
            state.ready.push_back((sample, bytes));
            self.changed.notify_all();
        }
    }

    fn next(&self) -> PyResult<Option<Sample>> {
        let state = self.state.lock().unwrap();
        let mut state = self
            .changed
            .wait_while(state, |state| {
                state.ready.is_empty() && !state.closed && !state.finished
            })
            .unwrap();
        if let Some((sample, bytes)) = state.ready.pop_front() {
            state.bytes -= bytes;
            self.changed.notify_all();
            Ok(Some(sample))
        } else if let Some(error) = state.error.take() {
            Err(error)
        } else {
            Ok(None)
        }
    }

    fn cancel(&self) {
        let mut state = self.state.lock().unwrap();
        state.closed = true;
        state.bytes = 0;
        let ready = std::mem::take(&mut state.ready);
        let error = state.error.take();
        self.changed.notify_all();
        drop(state);
        drop((ready, error));
    }
}

struct PythonSamples(Option<(Py<PyIterator>, Py<PyAny>)>);

impl PythonSamples {
    fn next(&self) -> PyResult<Option<(Sample, usize)>> {
        let (source, size) = self.0.as_ref().unwrap();
        Python::try_attach(|py| {
            let Some(sample) = source.bind(py).clone().next().transpose()? else {
                return Ok(None);
            };
            let bytes = size.call1(py, (&sample,))?.extract(py)?;
            Ok(Some((Sample::Parquet(sample.unbind()), bytes)))
        })
        .unwrap_or(Ok(None))
    }
}

impl Drop for PythonSamples {
    fn drop(&mut self) {
        let source = self.0.take();
        Python::try_attach(move |_| drop(source));
    }
}

#[pyclass(name = "_PrefetchIterator")]
pub(crate) struct PrefetchIterator {
    queue: Arc<Queue>,
    worker: Mutex<Option<JoinHandle<()>>>,
}

impl PrefetchIterator {
    pub(crate) fn start(
        max_bytes: usize,
        max_items: usize,
        produce: impl FnMut(usize, usize) -> PyResult<Option<(Sample, usize)>> + Send + 'static,
    ) -> PyResult<Self> {
        if max_bytes == 0 || max_items == 0 {
            return Err(PyValueError::new_err(
                "max_bytes and max_items must be positive",
            ));
        }
        let queue = Arc::new(Queue {
            state: Mutex::default(),
            changed: Condvar::new(),
            max_bytes,
            max_items,
        });
        let thread_queue = queue.clone();
        let worker = thread::Builder::new()
            .name("webshart-prefetch".into())
            .spawn(move || {
                let result = catch_unwind(AssertUnwindSafe(|| thread_queue.fill(produce)))
                    .unwrap_or_else(|_| Err(PyRuntimeError::new_err("Prefetch worker panicked")));
                let mut state = thread_queue.state.lock().unwrap();
                if !state.closed {
                    state.error = result.err();
                }
                state.finished = true;
                thread_queue.changed.notify_all();
            })
            .map_err(|err| PyOSError::new_err(err.to_string()))?;
        Ok(Self {
            queue,
            worker: Mutex::new(Some(worker)),
        })
    }
}

#[pymethods]
impl PrefetchIterator {
    #[staticmethod]
    fn _from_python(
        source: &Bound<'_, PyAny>,
        size: Py<PyAny>,
        max_bytes: usize,
        max_items: usize,
    ) -> PyResult<Self> {
        let samples = PythonSamples(Some((source.try_iter()?.unbind(), size)));
        Self::start(max_bytes, max_items, move |_, _| samples.next())
    }

    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __next__(&self, py: Python<'_>) -> PyResult<Option<Py<PyAny>>> {
        match py.detach(|| self.queue.next())? {
            Some(Sample::Tar(entry)) => Ok(Some(Py::new(py, entry)?.into_any())),
            Some(Sample::Parquet(entry)) => Ok(Some(entry)),
            None => Ok(None),
        }
    }

    fn close(&self, py: Python<'_>) {
        self.queue.cancel();
        let worker = self.worker.lock().unwrap().take();
        if let Some(worker) = worker {
            py.detach(|| {
                let _ = worker.join();
            });
        }
    }

    fn __enter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __exit__(
        &self,
        py: Python<'_>,
        _exc_type: &Bound<'_, PyAny>,
        _exc_value: &Bound<'_, PyAny>,
        _traceback: &Bound<'_, PyAny>,
    ) {
        self.close(py);
    }
}

impl Drop for PrefetchIterator {
    fn drop(&mut self) {
        self.queue.cancel();
    }
}
