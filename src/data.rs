use super::gamedb::serialization::Decoder;
use super::gamedb::game::{Game, Outcome};
use std::path::{PathBuf, Path};
use std::sync::{mpsc, Arc, Mutex};

use shakmaty::{Position, Chess, uci::Uci, File, Rank, Square, Color, Role};
use eyre::Result;
use pyo3::prelude::*;
use pyo3::types::PyList;
use pyo3::exceptions::PyValueError;

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
                    Color::White => {
                        match piece.role {
                            Role::Pawn => pos[x][y][0] = 1.0,
                            Role::Knight => pos[x][y][1] = 1.0,
                            Role::Bishop => pos[x][y][2] = 1.0,
                            Role::Rook => pos[x][y][3] = 1.0,
                            Role::Queen => pos[x][y][4] = 1.0,
                            Role::King => pos[x][y][5] = 1.0,
                        }
                    }
                    Color::Black => {
                        match piece.role {
                            Role::Pawn => pos[x][y][6] = 1.0,
                            Role::Knight => pos[x][y][7] = 1.0,
                            Role::Bishop => pos[x][y][8] = 1.0,
                            Role::Rook => pos[x][y][9] = 1.0,
                            Role::Queen => pos[x][y][10] = 1.0,
                            Role::King => pos[x][y][11] = 1.0,
                        }
                    }
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
    
        let uci = Uci::Normal { from: move_.move_from(), to: move_.move_to(), promotion: move_.promotion() };
        let move_ = uci.to_move(&board).unwrap();
        board = board.play(&move_).unwrap();
        positions.push(encode_position(&board));
    }

    (positions, encoded_out)
}

pub fn load_encoded_impl<'py>(py: Python<'py>, file_path: &Path, num_threads: usize) -> Result<(Bound<'py, PyList>, Bound<'py, PyList>)> {
    let decoder = Decoder::open(file_path)?;
    let games = decoder.raw_iter().collect::<Vec<_>>();
    let num_games = games.len();
    let games = Arc::new(Mutex::new(games));
    let encoded_ins = PyList::empty_bound(py);
    let encoded_outs = PyList::empty_bound(py);
    std::thread::scope(|scope| {
        let (tx, rx) = mpsc::channel();
        for _ in 0..num_threads.into() {
            let games = Arc::clone(&games);
            let tx = tx.clone();
            scope.spawn(move || {
                'outer: loop {
                    let work = games.lock().unwrap().pop();
                    match work {
                        Some(game) => {
                            let encoded = encode_game_positions(game);
                            tx.send(encoded).unwrap();
                        },
                        None => {
                            break 'outer;
                        }
                    }
                }
            });
        }
        for _ in 0..num_games {
            let (batch, outcome) = rx.recv().unwrap();
            for pos in batch {
                encoded_ins.append(PyArray::from_vec3_bound(py, &pos).unwrap()).unwrap();
                encoded_outs.append(outcome.clone()).unwrap();
            }
        }
    });
    Ok((encoded_ins, encoded_outs))
}   

#[pyfunction]
#[pyo3(pass_module)]
fn load_positions<'py>(module: &Bound<'py, PyModule>, file_path: &str, num_threads: usize) -> PyResult<(Bound<'py, PyList>, Bound<'py, PyList>)> {
    let file_path = PathBuf::from(file_path.to_string());
    load_encoded_impl(module.py(), &file_path, num_threads)
        .map_err(|e| PyValueError::new_err(format!("{:?}", e)))
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    let submodule = &PyModule::new_bound(m.py(), "data")?;

    submodule.add_function(wrap_pyfunction!(load_positions, submodule)?)?;

    //submodule.add_class::<Game>()?;
    //submodule.add_class::<GameLoader>()?;

    m.add_submodule(submodule)?;
    Ok(())
}