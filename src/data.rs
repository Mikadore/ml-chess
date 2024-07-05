use super::gamedb::game::{Game, Outcome};
use super::gamedb::serialization::Decoder;
use std::path::PathBuf;
use std::sync::{mpsc, Arc, Mutex};

use eyre::Result;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyList;
use shakmaty::{uci::Uci, Chess, Color, File, Position, Rank, Role, Square};
use std::{fs, io::BufReader};

use numpy::PyArray;

/// Model input layer is 8x8x13
pub type EncodedInput = Vec<Vec<Vec<f32>>>;

/// Model output layer is 3
pub type EncodedOutput = Vec<f32>;

pub fn encode_position(chess: &impl Position) -> EncodedInput {
    let mut pos = vec![vec![vec![0.0; 13]; 8]; 8];
    let board = chess.board();
    let turn_value = match chess.turn() {
        Color::Black => -1.0,
        Color::White => 1.0,
    };

    for x in 0..8 {
        for y in 0..8 {
            pos[x][y][12] = turn_value;
            let square = Square::from_coords(File::new(x as u32), Rank::new(7 - (y as u32)));
            if let Some(piece) = board.piece_at(square) {
                match piece.color {
                    Color::White => match piece.role {
                        Role::Pawn => pos[x][y][0] = 1.0,
                        Role::Knight => pos[x][y][1] = 1.0,
                        Role::Bishop => pos[x][y][2] = 1.0,
                        Role::Rook => pos[x][y][3] = 1.0,
                        Role::Queen => pos[x][y][4] = 1.0,
                        Role::King => pos[x][y][5] = 1.0,
                    },
                    Color::Black => match piece.role {
                        Role::Pawn => pos[x][y][6] = 1.0,
                        Role::Knight => pos[x][y][7] = 1.0,
                        Role::Bishop => pos[x][y][8] = 1.0,
                        Role::Rook => pos[x][y][9] = 1.0,
                        Role::Queen => pos[x][y][10] = 1.0,
                        Role::King => pos[x][y][11] = 1.0,
                    },
                }
            }
        }
    }
    pos
}

pub fn encode_game_positions(game: Vec<u8>) -> (Vec<EncodedInput>, EncodedOutput) {
    let game = bincode::deserialize::<Game>(&game).unwrap();
    let mut board = Chess::new();
    let mut positions = Vec::with_capacity(game.moves.len());

    let encoded_out = match game.outcome {
        Outcome::BlackWin => vec![0.0, 0.0, 1.0],
        Outcome::WhiteWin => vec![1.0, 0.0, 0.0],
        Outcome::Draw => vec![0.0, 1.0, 0.0],
    };

    for move_ in &game.moves {
        let uci = Uci::Normal {
            from: move_.move_from(),
            to: move_.move_to(),
            promotion: move_.promotion(),
        };
        let move_ = uci.to_move(&board).unwrap();
        board = board.play(&move_).unwrap();
        positions.push(encode_position(&board));
    }

    (positions, encoded_out)
}

pub fn load_encoded_impl<'py>(
    py: Python<'py>,
    decoder: &mut Decoder<BufReader<fs::File>>,
    max_games: usize,
    num_threads: usize,
) -> Result<(Bound<'py, PyList>, Bound<'py, PyList>)> {
    let mut games = Vec::with_capacity(max_games);
    for _ in 0..max_games {
        match decoder.read_game_raw().unwrap() {
            Some(g) => games.push(g),
            None => break,
        }
    }
    let num_games = games.len();
    let games = Arc::new(Mutex::new(games));
    let encoded_ins = PyList::empty_bound(py);
    let encoded_outs = PyList::empty_bound(py);
    std::thread::scope(|scope| {
        let (tx, rx) = mpsc::channel();
        for _ in 0..num_threads.into() {
            let games = Arc::clone(&games);
            let tx = tx.clone();
            scope.spawn(move || 'outer: loop {
                let work = games.lock().unwrap().pop();
                match work {
                    Some(game) => {
                        let encoded = encode_game_positions(game);
                        tx.send(encoded).unwrap();
                    }
                    None => {
                        break 'outer;
                    }
                }
            });
        }
        for _ in 0..num_games {
            let (batch, outcome) = rx.recv().unwrap();
            for pos in batch {
                encoded_ins
                    .append(PyArray::from_vec3_bound(py, &pos).unwrap())
                    .unwrap();
                encoded_outs.append(outcome.clone()).unwrap();
            }
        }
    });
    Ok((encoded_ins, encoded_outs))
}

#[pyclass]
pub struct PositionLoader {
    decoder: Decoder<BufReader<fs::File>>,
    num_threads: usize,
}

#[pymethods]
impl PositionLoader {
    #[new]
    pub fn new(filepath: &str, num_threads: usize) -> PyResult<Self> {
        (|| -> Result<Self> {
            let decoder = Decoder::open(&PathBuf::from(filepath.to_string()))?;
            Ok(Self {
                decoder,
                num_threads,
            })
        }())
        .map_err(|e| PyValueError::new_err(format!("{:?}", e)))
    }

    fn load_positions<'py>(
        mut slf: PyRefMut<'py, Self>,
        max_games: usize,
    ) -> PyResult<(Bound<'py, PyList>, Bound<'py, PyList>)> {
        let num_threads = slf.num_threads;
        load_encoded_impl(slf.py(), &mut slf.decoder, max_games, num_threads)
            .map_err(|e| PyValueError::new_err(format!("{:?}", e)))
    }
}

#[pyfunction]
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    let submodule = &PyModule::new_bound(m.py(), "data")?;

    submodule.add_class::<PositionLoader>()?;
    //submodule.add_class::<Game>()?;
    //submodule.add_class::<GameLoader>()?;

    m.add_submodule(submodule)?;
    Ok(())
}
