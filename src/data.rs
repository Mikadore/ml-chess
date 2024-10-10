use eyre::Result;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyBytesMethods};
use std::path::PathBuf;

use super::data;
use crate::games::game::{Game, Outcome};
use crate::games::serialization::Decoder;
use numpy::ndarray::{array, Array1, Array2, Array3, Array4, Axis};
use numpy::{PyArray2, PyArray4, PyArrayMethods};
use serde::{Deserialize, Serialize};
use shakmaty::{uci::UciMove, Chess, Color, File, Position, Rank, Role, Square};
use std::sync::{mpsc, Arc, Mutex};
use std::time::{Duration, Instant};
use tracing::{info, debug};

pub const FEATURES: usize = 37;
const F_W_PAWN: usize = 0;
const F_W_KNIGHT: usize = 1;
const F_W_BISHOP: usize = 2;
const F_W_ROOK: usize = 3;
const F_W_QUEEN: usize = 4;
const F_W_KING: usize = 5;
const F_B_PAWN: usize = 6;
const F_B_KNIGHT: usize = 7;
const F_B_BISHOP: usize = 8;
const F_B_ROOK: usize = 9;
const F_B_QUEEN: usize = 10;
const F_B_KING: usize = 11;
const F_TURN: usize = 12;
const F_ATTACKED_BY_W_PAWN: usize = 13;
const F_ATTACKED_BY_W_KNIGHT: usize = 14;
const F_ATTACKED_BY_W_BISHOP: usize = 15;
const F_ATTACKED_BY_W_ROOK: usize = 16;
const F_ATTACKED_BY_W_QUEEN: usize = 17;
const F_ATTACKED_BY_W_KING: usize = 18;
const F_ATTACKED_BY_B_PAWN: usize = 19;
const F_ATTACKED_BY_B_KNIGHT: usize = 20;
const F_ATTACKED_BY_B_BISHOP: usize = 21;
const F_ATTACKED_BY_B_ROOK: usize = 22;
const F_ATTACKED_BY_B_QUEEN: usize = 23;
const F_ATTACKED_BY_B_KING: usize = 24;
const F_ACCESSIBLE_BY_W_PAWN: usize = 25;
const F_ACCESSIBLE_BY_W_KNIGHT: usize = 26;
const F_ACCESSIBLE_BY_W_BISHOP: usize = 27;
const F_ACCESSIBLE_BY_W_ROOK: usize = 28;
const F_ACCESSIBLE_BY_W_QUEEN: usize = 29;
const F_ACCESSIBLE_BY_W_KING: usize = 30;
const F_ACCESSIBLE_BY_B_PAWN: usize = 31;
const F_ACCESSIBLE_BY_B_KNIGHT: usize = 32;
const F_ACCESSIBLE_BY_B_BISHOP: usize = 33;
const F_ACCESSIBLE_BY_B_ROOK: usize = 34;
const F_ACCESSIBLE_BY_B_QUEEN: usize = 35;
const F_ACCESSIBLE_BY_B_KING: usize = 36;

pub type NNInput = Array3<f32>;
pub type NNInputBatch = Array4<f32>;
pub type NNOutput = Array1<f32>;
pub type NNOutputBatch = Array2<f32>;

pub fn encode_position(chess: &impl Position) -> NNInput {
    let mut encoded = Array3::zeros((8, 8, FEATURES));
    let board = chess.board();
    let turn_value = match chess.turn() {
        Color::Black => -1.0,
        Color::White => 1.0,
    };

    for move_ in chess.legal_moves() {
        let piece = board.piece_at(move_.from().unwrap()).unwrap();
        let filter = match piece.color {
            Color::White => match piece.role {
                Role::Pawn => F_ACCESSIBLE_BY_W_PAWN,
                Role::Knight => F_ACCESSIBLE_BY_W_KNIGHT,
                Role::Bishop => F_ACCESSIBLE_BY_W_BISHOP,
                Role::Rook => F_ACCESSIBLE_BY_W_ROOK,
                Role::Queen => F_ACCESSIBLE_BY_W_QUEEN,
                Role::King => F_ACCESSIBLE_BY_W_KING,
            },
            Color::Black => match piece.role {
                Role::Pawn => F_ACCESSIBLE_BY_B_PAWN,
                Role::Knight => F_ACCESSIBLE_BY_B_KNIGHT,
                Role::Bishop => F_ACCESSIBLE_BY_B_BISHOP,
                Role::Rook => F_ACCESSIBLE_BY_B_ROOK,
                Role::Queen => F_ACCESSIBLE_BY_B_QUEEN,
                Role::King => F_ACCESSIBLE_BY_B_KING,
            },
        };
        let (x, y) = (
            move_.to().file().into(),
            7 - Into::<usize>::into(move_.to().rank()),
        );
        encoded[[x, y, filter]] = 1.0;
    }

    for x in 0..8 {
        for y in 0..8 {
            encoded[[x, y, F_TURN]] = turn_value;
            let square = Square::from_coords(File::new(x as u32), Rank::new(7 - (y as u32)));
            if let Some(piece) = board.piece_at(square) {
                match piece.color {
                    Color::White => match piece.role {
                        Role::Pawn => encoded[[x, y, F_W_PAWN]] = 1.0,
                        Role::Knight => encoded[[x, y, F_W_KNIGHT]] = 1.0,
                        Role::Bishop => encoded[[x, y, F_W_BISHOP]] = 1.0,
                        Role::Rook => encoded[[x, y, F_W_ROOK]] = 1.0,
                        Role::Queen => encoded[[x, y, F_W_QUEEN]] = 1.0,
                        Role::King => encoded[[x, y, F_W_KING]] = 1.0,
                    },
                    Color::Black => match piece.role {
                        Role::Pawn => encoded[[x, y, F_B_PAWN]] = 1.0,
                        Role::Knight => encoded[[x, y, F_B_KNIGHT]] = 1.0,
                        Role::Bishop => encoded[[x, y, F_B_BISHOP]] = 1.0,
                        Role::Rook => encoded[[x, y, F_B_ROOK]] = 1.0,
                        Role::Queen => encoded[[x, y, F_B_QUEEN]] = 1.0,
                        Role::King => encoded[[x, y, F_B_KING]] = 1.0,
                    },
                }
                let mut attacks_bb = board.attacks_from(square);
                let attack_filter = match piece.color {
                    Color::White => match piece.role {
                        Role::Pawn => F_ATTACKED_BY_W_PAWN,
                        Role::Knight => F_ATTACKED_BY_W_KNIGHT,
                        Role::Bishop => F_ATTACKED_BY_W_BISHOP,
                        Role::Rook => F_ATTACKED_BY_W_ROOK,
                        Role::Queen => F_ATTACKED_BY_W_QUEEN,
                        Role::King => F_ATTACKED_BY_W_KING,
                    },
                    Color::Black => match piece.role {
                        Role::Pawn => F_ATTACKED_BY_B_PAWN,
                        Role::Knight => F_ATTACKED_BY_B_KNIGHT,
                        Role::Bishop => F_ATTACKED_BY_B_BISHOP,
                        Role::Rook => F_ATTACKED_BY_B_ROOK,
                        Role::Queen => F_ATTACKED_BY_B_QUEEN,
                        Role::King => F_ATTACKED_BY_B_KING,
                    },
                };
                while let Some(sq) = attacks_bb.pop_front() {
                    let (x, y) = (sq.file().into(), 7 - Into::<usize>::into(sq.rank()));
                    encoded[[x, y, attack_filter]] = 1.0;
                }
            }
        }
    }
    encoded
}

pub fn encode_outcome(outcome: Outcome) -> NNOutput {
    match outcome {
        Outcome::BlackWin => array![0.0, 0.0, 1.0],
        Outcome::WhiteWin => array![1.0, 0.0, 0.0],
        Outcome::Draw => array![0.0, 1.0, 0.0],
    }
}

#[pyclass]
pub struct TrainData {
    ins: Py<PyArray4<f32>>,
    outs: Py<PyArray2<f32>>,
}

#[derive(Serialize, Deserialize)]
struct TrainDataFileHeader {
    magic: [u8; 16],
    ins_shape: [usize; 4],
    outs_shape: [usize; 2],
}

impl TrainData {
    pub fn from_games(games: Vec<Vec<u8>>) -> (NNInputBatch, NNOutputBatch) {
        let num_games = games.len();
        let games = Arc::new(Mutex::new(games));
        let mut inputs = Array4::zeros((0, 8, 8, data::FEATURES));
        let mut outputs = Array2::zeros((0, 3));
        std::thread::scope(|scope| {
            let (tx, rx) = mpsc::channel();
            for _ in 0..std::thread::available_parallelism()
                .map(|n| n.get())
                .unwrap_or(1)
            {
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
                    inputs.push(Axis(0), pos.view()).unwrap();
                    outputs.push(Axis(0), outcome.view()).unwrap();
                }
            }
        });
        info!(
            "Processed {} games into {} positions",
            num_games,
            inputs.shape()[0]
        );

        let x_bytes = inputs.len() * std::mem::size_of::<f32>();
        let y_bytes = outputs.len() * std::mem::size_of::<f32>();

        info!(
            "Total memory of encoded positions: {:.2} Mb",
            (x_bytes + y_bytes) as f64 / (1024.0 * 1024.0)
        );

        (inputs, outputs)
    }

    pub fn from_games_py(py: Python<'_>, games: Vec<Vec<u8>>) -> Self {
        let (ins, outs) = Self::from_games(games);
        let ins = PyArray4::from_owned_array_bound(py, ins).unbind();
        let outs = PyArray2::from_owned_array_bound(py, outs).unbind();

        TrainData { ins, outs }
    }

    pub fn encode_bin(ins: NNInputBatch, outs: NNOutputBatch) -> Vec<u8> {
        assert!(ins.shape().len() == 4);
        assert!(outs.shape().len() == 2);

        let header = TrainDataFileHeader {
            magic: *b"mychesstraindata",
            ins_shape: ins.shape().try_into().unwrap(),
            outs_shape: outs.shape().try_into().unwrap(),
        };
        let mut header = bincode::serialize(&header).unwrap();

        let ins = ins.as_slice().unwrap();
        let outs = outs.as_slice().unwrap();

        let mut data = Vec::with_capacity(
            ins.len() * std::mem::size_of::<f32>() + outs.len() * std::mem::size_of::<f32>(),
        );
        for &f in ins {
            data.extend_from_slice(&f.to_le_bytes());
        }

        for &f in outs {
            data.extend_from_slice(&f.to_le_bytes());
        }

        let mut data = zstd::stream::encode_all(data.as_slice(), 0).unwrap();

        let mut encoded= header.len().to_le_bytes().to_vec();
        encoded.append(&mut header);
        encoded.append(&mut data);

        encoded
    }

    pub fn encode_bin_py(&self, py: Python<'_>) -> Vec<u8> {
        let ins = self.ins.bind(py).readonly().as_array().to_owned();
        let outs = self.outs.bind(py).readonly().as_array().to_owned();
        Self::encode_bin(ins, outs)
    }

    fn bytes_to_floats(data: &[u8]) -> Vec<f32> {
        assert!(data.len() % std::mem::size_of::<f32>() == 0);
        data.chunks_exact(std::mem::size_of::<f32>())
            .map(TryInto::<[u8; 4]>::try_into)
            .map(Result::unwrap)
            .map(f32::from_le_bytes)
            .collect()
    }

    pub fn decode_bin(data: &[u8]) -> (NNInputBatch, NNOutputBatch) {
        let header_offset = std::mem::size_of::<usize>();
        let header_size = usize::from_le_bytes(data[..header_offset].try_into().unwrap());
        let data_offset = header_offset + header_size;
        let header =
            bincode::deserialize::<TrainDataFileHeader>(&data[header_offset..data_offset]).unwrap();
        let data = &data[data_offset..];
        let data = zstd::stream::decode_all(data).unwrap();

        assert!(header.magic == *b"mychesstraindata");
        assert_eq!(header.ins_shape[1], 8);
        assert_eq!(header.ins_shape[2], 8);
        assert_eq!(header.ins_shape[3], data::FEATURES);
        assert_eq!(header.outs_shape[1], 3);

        let ins_bytes = header.ins_shape.iter().product::<usize>() * std::mem::size_of::<f32>();
        let outs_bytes = header.outs_shape.iter().product::<usize>() * std::mem::size_of::<f32>();

        assert_eq!(data.len(), ins_bytes + outs_bytes);

        let ins = Self::bytes_to_floats(&data[..ins_bytes]);
        let ins = Array4::from_shape_vec(header.ins_shape, ins).unwrap();

        let outs = Self::bytes_to_floats(&data[ins_bytes..]);
        let outs = Array2::from_shape_vec(header.outs_shape, outs).unwrap();
        (ins, outs)
    }

    pub fn decode_bin_py(py: Python<'_>, data: &[u8]) -> Self {
        let (ins, outs) = Self::decode_bin(data);
        let ins = PyArray4::from_owned_array_bound(py, ins).unbind();
        let outs = PyArray2::from_owned_array_bound(py, outs).unbind();

        TrainData { ins, outs }
    }
}

#[pymethods]
impl TrainData {
    fn get_ins(slf: PyRef<'_, Self>) -> Bound<'_, PyArray4<f32>> {
        let ins = slf.ins.clone_ref(slf.py());
        ins.into_bound(slf.py())
    }

    fn get_outs(slf: PyRef<'_, Self>) -> Bound<'_, PyArray2<f32>> {
        let outs = slf.outs.clone_ref(slf.py());
        outs.into_bound(slf.py())
    }

    fn to_bytes(slf: PyRef<'_, Self>) -> PyResult<Bound<'_, PyBytes>> {
        let data = slf.encode_bin_py(slf.py());
        PyBytes::new_bound_with(slf.py(), data.len(), |buf| {
            buf.copy_from_slice(&data);
            Ok(())
        })
    }

    #[staticmethod]
    fn from_bytes(py: Python<'_>, data: &Bound<'_, PyBytes>) -> Self {
        Self::decode_bin_py(py, data.as_bytes())
    }

    fn save(slf: PyRef<'_, Self>, path: &str) -> PyResult<()> {
        let path = PathBuf::from(path);
        std::fs::write(path, slf.encode_bin_py(slf.py()))
            .map_err(|e| PyValueError::new_err(format!("{:#?}", e)))?;
        Ok(())
    }

    #[staticmethod]
    fn load(py: Python<'_>, path: &str) -> PyResult<Self> {
        let path = PathBuf::from(path);
        let data = std::fs::read(path).map_err(|e| PyValueError::new_err(format!("{:#?}", e)))?;
        Ok(Self::decode_bin_py(py, &data))
    }

    #[staticmethod]
    fn convert_games_and_save<'py>(_py: Python<'_>, path: PathBuf, max_games: usize, name: &str) -> PyResult<()> {
        let games: Vec<Vec<u8>> = Decoder::open(&path).unwrap().raw_iter().collect();
        let name = name.to_string();
        let mut handles = Vec::with_capacity(games.len());
        for (i, batch) in games.chunks(max_games).enumerate() {
            let data = data::TrainData::from_games(batch.to_vec());
            let path = format!("data/train/{name}/{i:03}.bin");
            handles.push(std::thread::spawn(move || {
                let data = data::TrainData::encode_bin(data.0, data.1);
                std::fs::write(path, data).unwrap();
            }));
        }
        info!("Waiting for files to finish saving");
        for h in handles {
            h.join().unwrap();
        }
        Ok(())
    }
}

#[derive(Clone, Copy)]
struct ReadSend {
    read: usize,
    sent: usize,
}

#[pyclass]
pub struct TrainDataLoader {
    receiver: mpsc::Receiver<(NNInputBatch, NNOutputBatch)>,
    // This is a kinda hacky way to inform the worker thread
    // when we want more data. CBA with a condvar right now
    read_sent_count: Arc<Mutex<ReadSend>>,
}

#[pymethods]
impl TrainDataLoader {
    #[new]
    fn new(files: Vec<PathBuf>, prefetch: usize) -> Self {
        let (tx, receiver) = mpsc::channel();
        let read_sent_count = Arc::new(Mutex::new(ReadSend { read: 0, sent: 0 }));
        {
            let files = Arc::new(Mutex::new(files));
            for _ in 0..prefetch.max(1).min(5) {
                let files = Arc::clone(&files);
                let read_sent_count = Arc::clone(&read_sent_count);
                let tx = tx.clone();
                std::thread::spawn(move || {
                    loop {
                        if files.lock().unwrap().is_empty() {
                            break
                        }
                        let ReadSend { read, sent } = {
                            let mg = read_sent_count.lock().unwrap();
                            *mg
                        };
                        if read > sent || (sent - read) < prefetch {
                            let file = {
                                let mut mg = files.lock().unwrap();
                                match mg.pop() {
                                    Some(f) => f,
                                    None => break,
                                }
                            };
                            let data = std::fs::read(&file).unwrap();
                            let start = Instant::now();
                            let batch = TrainData::decode_bin(&data);
                            let delta = start.elapsed().as_millis() as f64 / 1000.0;
                            info!("Loaded {} in {:.2} seconds", file.file_name().unwrap().to_str().unwrap(), delta);
                            tx.send(batch).unwrap();
                            {
                                let mut mg = read_sent_count.lock().unwrap();
                                mg.sent += 1;
                            }
                        } else {
                            std::thread::sleep(Duration::from_millis(100));
                        }
                    }
                });
            }
        }
        Self { receiver, read_sent_count }
    }

    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __next__(slf: PyRefMut<'_, Self>) -> Option<TrainData> {
        debug!("Reading next batch");
        let now = Instant::now();
        {
            let mut mg = slf.read_sent_count.lock().unwrap();
            debug!("Read {} batches total. Sent {} batches total", mg.read, mg.sent);
            mg.read += 1;
        }
        match slf.receiver.recv() {
            Ok(batch) => {
                debug!("Read batch in {:.2}s", now.elapsed().as_secs() as f64 / 1000.0);
                Some(TrainData {
                    ins: PyArray4::from_owned_array_bound(slf.py(), batch.0).unbind(),
                    outs: PyArray2::from_owned_array_bound(slf.py(), batch.1).unbind(),
                })
            },
            Err(_) => None,
        }
    }
}



pub fn encode_game_positions(game: Vec<u8>) -> (Vec<NNInput>, NNOutput) {
    let game = bincode::deserialize::<Game>(&game).unwrap();
    let mut board = Chess::new();
    let mut positions = Vec::with_capacity(game.moves.len());

    for move_ in &game.moves {
        let uci = UciMove::Normal {
            from: move_.move_from(),
            to: move_.move_to(),
            promotion: move_.promotion(),
        };
        let move_ = uci.to_move(&board).unwrap();
        board = board.play(&move_).unwrap();
        positions.push(encode_position(&board));
    }

    (positions, encode_outcome(game.outcome))
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<TrainData>()?;
    m.add_class::<TrainDataLoader>()?;
    Ok(())
}
