use eyre::Result;
use serde::{Deserialize, Serialize};
use shakmaty::{san::San, uci::Uci, Chess, Position};
use std::io::Write;

/// Bit field representing a UCI move:
/// the MSB is 0
/// bits [14:12] indicate promotion:
///     - 0 = none
///     - 1 = Pawn
///     - 2 = Knight
///     - 3 = Bishop
///     - 4 = Rook
///     - 5 = Queen
///     - 6 = King
///
/// bits [11:6] are the from encoding:
///     - bits [11:9] are the file (a=0, ..)
///     - bits [ 8:6] are the rank
/// bits [ 5:0] are the to encoding:
///     - bits [5:3] are the file (a=0, ..)
///     - bits [2:0] are the rank
#[derive(Serialize, Deserialize, Clone, Copy, PartialEq, Eq)]
pub struct Move {
    pub(crate) bitfield: u16,
}

impl std::fmt::Debug for Move {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}{}", self.move_from(), self.move_to())?;
        if let Some(promotion) = self.promotion() {
            write!(
                f,
                "={}",
                match promotion {
                    shakmaty::Role::Bishop => "B",
                    shakmaty::Role::King => "K",
                    shakmaty::Role::Knight => "N",
                    shakmaty::Role::Pawn => "P",
                    shakmaty::Role::Queen => "Q",
                    shakmaty::Role::Rook => "R",
                }
            )?;
        }
        Ok(())
    }
}

impl Move {
    pub fn new(
        from: shakmaty::Square,
        to: shakmaty::Square,
        promotion: Option<shakmaty::Role>,
    ) -> Self {
        let from = (file_to_int(from.file()) << 3 | rank_to_int(from.rank())) as u16;
        let to = (file_to_int(to.file()) << 3 | rank_to_int(to.rank())) as u16;
        let promotion = match promotion {
            None => 0u16,
            Some(shakmaty::Role::Pawn) => 1,
            Some(shakmaty::Role::Knight) => 2,
            Some(shakmaty::Role::Bishop) => 3,
            Some(shakmaty::Role::Rook) => 4,
            Some(shakmaty::Role::Queen) => 5,
            Some(shakmaty::Role::King) => 6,
        };
        let bitfield = (promotion << 12) | (from << 6) | to;
        Self { bitfield }
    }

    pub fn promotion(&self) -> Option<shakmaty::Role> {
        let promotion = (self.bitfield & (0b111 << 12)) >> 12;
        match promotion {
            0 => None,
            1 => Some(shakmaty::Role::Pawn),
            2 => Some(shakmaty::Role::Knight),
            3 => Some(shakmaty::Role::Bishop),
            4 => Some(shakmaty::Role::Rook),
            5 => Some(shakmaty::Role::Queen),
            6 => Some(shakmaty::Role::King),
            _ => unreachable!(),
        }
    }

    pub fn move_from(&self) -> shakmaty::Square {
        let from_file = (self.bitfield & (0b111 << 9)) >> 9;
        let from_rank = (self.bitfield & (0b111 << 6)) >> 6;
        let file = file_from_int(from_file as u8);
        let rank = rank_from_int(from_rank as u8);
        shakmaty::Square::from_coords(file, rank)
    }

    pub fn move_to(&self) -> shakmaty::Square {
        let to_file = (self.bitfield & (0b111 << 3)) >> 3;
        let to_rank = self.bitfield & 0b111;
        let file = file_from_int(to_file as u8);
        let rank = rank_from_int(to_rank as u8);
        shakmaty::Square::from_coords(file, rank)
    }

    pub fn to_uci(&self) -> String {
        format!("{}{}{}", self.move_from(), self.move_to(), match self.promotion() {
            Some(piece) => match piece {
                shakmaty::Role::Bishop => "b",
                shakmaty::Role::King => "k",
                shakmaty::Role::Knight => "n",
                shakmaty::Role::Pawn => "p",
                shakmaty::Role::Queen => "q",
                shakmaty::Role::Rook => "r",
            },
            None => "",
        })
    }
}

fn file_from_int(x: u8) -> shakmaty::File {
    match x {
        0 => shakmaty::File::A,
        1 => shakmaty::File::B,
        2 => shakmaty::File::C,
        3 => shakmaty::File::D,
        4 => shakmaty::File::E,
        5 => shakmaty::File::F,
        6 => shakmaty::File::G,
        7 => shakmaty::File::H,
        _ => unreachable!(),
    }
}

fn rank_from_int(x: u8) -> shakmaty::Rank {
    match x {
        0 => shakmaty::Rank::First,
        1 => shakmaty::Rank::Second,
        2 => shakmaty::Rank::Third,
        3 => shakmaty::Rank::Fourth,
        4 => shakmaty::Rank::Fifth,
        5 => shakmaty::Rank::Sixth,
        6 => shakmaty::Rank::Seventh,
        7 => shakmaty::Rank::Eighth,
        _ => unreachable!(),
    }
}

fn file_to_int(f: shakmaty::File) -> u8 {
    match f {
        shakmaty::File::A => 0,
        shakmaty::File::B => 1,
        shakmaty::File::C => 2,
        shakmaty::File::D => 3,
        shakmaty::File::E => 4,
        shakmaty::File::F => 5,
        shakmaty::File::G => 6,
        shakmaty::File::H => 7,
    }
}

fn rank_to_int(r: shakmaty::Rank) -> u8 {
    match r {
        shakmaty::Rank::First => 0,
        shakmaty::Rank::Second => 1,
        shakmaty::Rank::Third => 2,
        shakmaty::Rank::Fourth => 3,
        shakmaty::Rank::Fifth => 4,
        shakmaty::Rank::Sixth => 5,
        shakmaty::Rank::Seventh => 6,
        shakmaty::Rank::Eighth => 7,
    }
}

#[test]
fn test_move_identity() {
    let promotion_all = [
        None,
        Some(shakmaty::Role::Pawn),
        Some(shakmaty::Role::Knight),
        Some(shakmaty::Role::Bishop),
        Some(shakmaty::Role::Rook),
        Some(shakmaty::Role::Queen),
        Some(shakmaty::Role::King),
    ];
    for promotion in promotion_all {
        for from in shakmaty::Square::ALL {
            for to in shakmaty::Square::ALL {
                let m = Move::new(from, to, promotion);
                assert_eq!(m.move_from(), from);
                assert_eq!(m.move_to(), to);
                assert_eq!(m.promotion(), promotion);
            }
        }
    }
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Eq)]
pub enum Outcome {
    WhiteWin,
    BlackWin,
    Draw,
}

impl std::fmt::Display for Outcome {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", match self {
            Self::WhiteWin => "1-0",
            Self::BlackWin => "0-1",
            Self::Draw => "1/2-1/2",
        })
    }
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Eq)]
pub struct Game {
    pub white_name: String,
    pub black_name: String,
    pub white_elo: i32,
    pub black_elo: i32,
    pub outcome: Outcome,
    pub timectl_sec: i32,
    pub timectl_inc: i32,
    pub moves: Vec<Move>,
}

impl Game {
    pub fn write_pgn(&self, mut w: impl Write) -> Result<()> {
        writeln!(w, "[White \"{}\"]", self.white_name)?;
        writeln!(w, "[WhiteElo \"{}\"]", self.white_elo)?;
        writeln!(w, "[Black \"{}\"]", self.black_name)?;
        writeln!(w, "[BlackElo \"{}\"]", self.black_elo)?;
        writeln!(
            w,
            "[TimeControl \"{}+{}\"]",
            self.timectl_sec, self.timectl_inc
        )?;
        writeln!(w, "[Termination \"Normal\"]")?;
        writeln!(w, "")?;

        let mut pos = Chess::new();

        for (i, move_pair) in self.moves.chunks(2).enumerate() {
            write!(w, "{}. ", i + 1)?;

            let mut move_pair = move_pair.iter();
            let white_move = move_pair.next().unwrap();
            let white_move = Uci::Normal {
                from: white_move.move_from(),
                to: white_move.move_to(),
                promotion: white_move.promotion(),
            };
            let white_move = white_move.to_move(&pos).unwrap();
            let white_san = San::from_move(&pos, &white_move);

            write!(w, "{} ", white_san)?;
            pos = pos.play(&white_move).unwrap();

            if let Some(black_move) = move_pair.next() {
                let black_move = Uci::Normal {
                    from: black_move.move_from(),
                    to: black_move.move_to(),
                    promotion: black_move.promotion(),
                };
                let black_move = black_move.to_move(&pos)?;
                let black_san = San::from_move(&pos, &black_move);
                write!(w, "{} ", black_san)?;
                pos = pos.play(&black_move).unwrap();
            }
        }

        writeln!(w, "{}", self.outcome)?;
        writeln!(w)?;

        Ok(())
    }
}
