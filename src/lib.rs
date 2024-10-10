use pyo3::prelude::*;

mod games;
mod data;

#[pymodule]
fn chessers(m: &Bound<'_, PyModule>) -> PyResult<()> {
    tracing_subscriber::fmt::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .init();

    games::register(m)?;
    data::register(m)?;
    Ok(())
}