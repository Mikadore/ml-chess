use pyo3::prelude::*;

mod gamedb;
mod data;

#[pymodule]
fn chessers(m: &Bound<'_, PyModule>) -> PyResult<()> {
    gamedb::register(m)?;
    data::register(m)?;
    Ok(())
}