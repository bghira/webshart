use crate::jxl_codec::{decode, Pixels};
use pyo3::exceptions::PyValueError;
use pyo3::ffi;
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyMemoryView};
use std::ffi::c_int;

const CODESTREAM_SIGNATURE: &[u8] = b"\xff\x0a";
const CONTAINER_SIGNATURE: &[u8] = b"\x00\x00\x00\x0cJXL \x0d\x0a\x87\x0a";

/// Interleaved 8-bit pixels and metadata from the first display frame.
#[pyclass(module = "webshart", frozen, skip_from_py_object)]
pub struct DecodedJxlImage {
    #[pyo3(get)]
    width: u32,
    #[pyo3(get)]
    height: u32,
    #[pyo3(get)]
    mode: &'static str,
    pixels: Vec<u8>,
    #[pyo3(get)]
    exif: Option<Py<PyBytes>>,
    #[pyo3(get)]
    icc_profile: Py<PyBytes>,
}

#[pymethods]
impl DecodedJxlImage {
    #[getter]
    fn pixels<'py>(slf: Bound<'py, Self>) -> PyResult<Bound<'py, PyMemoryView>> {
        PyMemoryView::from(slf.as_any())
    }

    unsafe fn __getbuffer__(
        slf: Bound<'_, Self>,
        view: *mut ffi::Py_buffer,
        flags: c_int,
    ) -> PyResult<()> {
        let pixels = &slf.get().pixels;
        // FillInfo holds a reference to the frozen owner. The Vec stays alive and
        // cannot change while Python holds the read-only buffer.
        let result = unsafe {
            ffi::PyBuffer_FillInfo(
                view,
                slf.as_ptr(),
                pixels.as_ptr().cast_mut().cast(),
                pixels.len() as isize,
                1,
                flags,
            )
        };
        if result == 0 {
            Ok(())
        } else {
            Err(PyErr::fetch(slf.py()))
        }
    }
}

/// The decoder error for one JPEG XL payload.
#[pyclass(module = "webshart", frozen, skip_from_py_object)]
pub struct JxlDecodeError {
    #[pyo3(get)]
    message: String,
}

#[derive(Clone, Copy)]
pub(crate) struct JxlOptions {
    max_pixels: usize,
    allocation_limit: usize,
}

impl JxlOptions {
    pub(crate) fn new(max_pixels: usize) -> PyResult<Self> {
        let allocation_limit = max_pixels
            .checked_mul(32)
            .filter(|limit| *limit > 0)
            .ok_or_else(|| PyValueError::new_err("invalid JPEG XL pixel limit"))?
            .max(1024 * 1024);
        Ok(Self {
            max_pixels,
            allocation_limit,
        })
    }

    pub(crate) fn decode(self, payload: &[u8]) -> Result<Pixels, String> {
        decode(payload, self.max_pixels, self.allocation_limit)
    }
}

pub(crate) fn is_jxl(payload: &[u8]) -> bool {
    payload.starts_with(CODESTREAM_SIGNATURE) || payload.starts_with(CONTAINER_SIGNATURE)
}

pub(crate) fn decoded_to_python(
    py: Python<'_>,
    decoded: Result<Pixels, String>,
) -> PyResult<Py<PyAny>> {
    match decoded {
        Err(message) => Ok(Py::new(py, JxlDecodeError { message })?.into_any()),
        Ok(image) => Ok(Py::new(
            py,
            DecodedJxlImage {
                width: image.width,
                height: image.height,
                mode: image.mode,
                pixels: image.data,
                exif: image
                    .exif
                    .as_deref()
                    .map(|data| PyBytes::new(py, data).unbind()),
                icc_profile: PyBytes::new(py, &image.icc_profile).unbind(),
            },
        )?
        .into_any()),
    }
}
