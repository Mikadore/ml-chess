use pyo3::prelude::*;

mod games;
mod data;

#[pymodule]
fn chessers(m: &Bound<'_, PyModule>) -> PyResult<()> {
    games::register(m)?;
    data::register(m)?;
    Ok(())
}