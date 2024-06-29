use pyo3::prelude::*;

mod gamedb;

#[pymodule]
fn chessers(m: &Bound<'_, PyModule>) -> PyResult<()> {
    gamedb::register(m)?;
    Ok(())
}