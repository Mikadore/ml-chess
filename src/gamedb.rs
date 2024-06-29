use eyre::{Context, Result};
use pgn_reader::BufferedReader;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use std::fs::{File, OpenOptions};
use std::io::{BufReader, BufWriter};
use std::path::{Path, PathBuf};

mod data;
mod pgn;
mod serialization;

fn pgn_to_bin_impl(
    pgn_path: &Path,
    bin_path: &Path,
    min_elo: i32,
    max_elo_diff: i32,
) -> Result<()> {
    let f = File::open(pgn_path)?;
    let reader = BufReader::new(f);
    let mut reader = BufferedReader::new(reader);
    let mut writer = serialization::Encoder::open(bin_path)?;
    let mut visitor = pgn::PgnVisitor::new();
    while let Some(game) = reader.read_game(&mut visitor)? {
        if let Some(game) = game? {
            if ((game.black_elo - game.white_elo).abs() <= max_elo_diff)
                && (game.black_elo > min_elo)
                && (game.white_elo > min_elo)
            {
                writer.write_game(&game)?;
            }
        }
    }
    Ok(())
}

#[pyfunction]
fn pgn_to_bin(pgn_path: &str, bin_path: &str, min_elo: i32, max_elo_diff: i32) -> PyResult<()> {
    let pgn_path = PathBuf::from(pgn_path.to_string());
    let bin_path = PathBuf::from(bin_path.to_string());
    pgn_to_bin_impl(&pgn_path, &bin_path, min_elo, max_elo_diff)
        .map_err(|e| PyValueError::new_err(format!("{:?}", e)))
}

fn bin_to_pgn_impl(
    bin_path: &Path,
    pgn_path: &Path,
    min_elo: i32,
    max_elo_diff: i32,
) -> Result<()> {
    let f = OpenOptions::new()
        .write(true)
        .truncate(true)
        .create(true)
        .open(pgn_path)
        .wrap_err("failed to open target pgn file")?;

    let mut writer = BufWriter::new(f);
    let mut decoder =
        serialization::Decoder::open(bin_path).wrap_err("failed to open source bin file")?;

    while let Some(game) = decoder.read_game()? {
        if ((game.black_elo - game.white_elo).abs() <= max_elo_diff)
            && (game.black_elo > min_elo)
            && (game.white_elo > min_elo)
        {
            game.write_pgn(&mut writer)?;
        }
    }
    Ok(())
}

#[pyfunction]
fn bin_to_pgn(bin_path: &str, pgn_path: &str, min_elo: i32, max_elo_diff: i32) -> PyResult<()> {
    let pgn_path = PathBuf::from(pgn_path.to_string());
    let bin_path = PathBuf::from(bin_path.to_string());
    bin_to_pgn_impl(&bin_path, &pgn_path, min_elo, max_elo_diff)
        .map_err(|e| PyValueError::new_err(format!("{:?}", e)))
}

#[pyclass]
#[derive(Debug, Clone, PartialEq, Eq)]
struct Game {
    inner: data::Game,
    outcome: String,
}

impl Game {
    fn new(inner: data::Game) -> Self {
        Self {
            outcome: match inner.outcome {
                data::Outcome::BlackWin => "0-1".to_string(),
                data::Outcome::WhiteWin => "1-0".to_string(),
                data::Outcome::Draw => "1/2-1/2".to_string(),
            },
            inner,
        }
    }
}

#[pymethods]
impl Game {
    fn white(&self) -> &str {
        &self.inner.white_name
    }

    fn black(&self) -> &str {
        &self.inner.black_name
    }

    fn white_elo(&self) -> i32 {
        self.inner.white_elo
    }

    fn black_elo(&self) -> i32 {
        self.inner.black_elo
    }

    fn outcome(&self) -> &str {
        &self.outcome
    }

    fn timectl_sec(&self) -> i32 {
        self.inner.timectl_sec
    }

    fn timectl_inc(&self) -> i32 {
        self.inner.timectl_inc
    }

    fn moves(&self) -> Vec<String> {
        self.inner.moves.iter().map(|m| m.to_uci()).collect()
    }
}

#[pyclass]
#[derive(Debug)]
struct GameLoader {
    decoder: serialization::Decoder<BufReader<File>>,
}

#[pymethods]
impl GameLoader {
    #[new]
    fn new(file_path: &str) -> PyResult<Self> {
        (|| -> Result<Self> {
            let decoder = serialization::Decoder::open(&PathBuf::from(file_path))?;
            Ok(Self { decoder })
        }())
        .map_err(|e| PyValueError::new_err(format!("{:?}", e)))
    }

    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __next__(mut slf: PyRefMut<'_, Self>) -> Option<PyResult<Game>> {
        match slf.decoder.read_game() {
            Err(e) => Some(Err(PyValueError::new_err(format!("{:?}", e)))),
            Ok(Some(g)) => Some(Ok(Game::new(g))),
            Ok(None) => None,
        }
    }
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    let submodule = &PyModule::new_bound(m.py(), "pgn")?;

    submodule.add_function(wrap_pyfunction!(pgn_to_bin, submodule)?)?;
    submodule.add_function(wrap_pyfunction!(bin_to_pgn, submodule)?)?;

    submodule.add_class::<Game>()?;
    submodule.add_class::<GameLoader>()?;

    m.add_submodule(submodule)?;
    Ok(())
}
