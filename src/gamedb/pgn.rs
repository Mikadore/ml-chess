use super::data::*;
use eyre::{ensure, Result};
use pgn_reader::{SanPlus, Skip, Visitor};
use shakmaty::{Chess, Position};

#[derive(Debug, Clone)]
struct Headers {
    white_name: Option<String>,
    black_name: Option<String>,
    white_elo: Option<i32>,
    black_elo: Option<i32>,
    outcome: Option<Outcome>,
    timectl_sec: Option<i32>,
    timectl_inc: Option<i32>,
}

impl Headers {
    fn empty() -> Self {
        Self {
            white_name: None,
            black_name: None,
            white_elo: None,
            black_elo: None,
            outcome: None,
            timectl_sec: None,
            timectl_inc: None,
        }
    }
}

#[derive(Debug, Clone)]
pub struct PgnVisitor {
    headers: Headers,
    moves: Vec<Move>,
    board: Chess,
    skip: bool,
}

impl PgnVisitor {
    pub fn new() -> Self {
        Self {
            headers: Headers::empty(),
            moves: vec![],
            skip: false,
            board: Chess::new(),
        }
    }
}

impl Visitor for PgnVisitor {
    type Result = Result<Option<Game>>;

    fn begin_game(&mut self) {
        self.headers = Headers::empty();
        self.moves = Vec::new();
        self.skip = false;
        self.board = Chess::new();
    }

    fn end_game(&mut self) -> Self::Result {
        if self.skip {
            return Ok(None);
        }

        ensure!(
            self.headers.black_name.is_some(),
            "Black player name is missing"
        );
        let black_name = self.headers.black_name.as_ref().unwrap().to_owned();

        ensure!(
            self.headers.white_name.is_some(),
            "White player name is missing"
        );
        let white_name = self.headers.white_name.as_ref().unwrap().to_owned();

        ensure!(
            self.headers.black_elo.is_some(),
            "Black player elo is missing"
        );
        let black_elo = *self.headers.black_elo.as_ref().unwrap();

        ensure!(
            self.headers.white_elo.is_some(),
            "White player elo is missing"
        );
        let white_elo = *self.headers.white_elo.as_ref().unwrap();

        ensure!(self.headers.timectl_sec.is_some(), "Timecontrol missing");
        let timectl_sec = *self.headers.timectl_sec.as_ref().unwrap();

        ensure!(self.headers.timectl_inc.is_some(), "Timecontrol missing");
        let timectl_inc = *self.headers.timectl_inc.as_ref().unwrap();

        ensure!(self.headers.outcome.is_some(), "Outcome is missing");
        let outcome = self.headers.outcome.as_ref().unwrap().to_owned();

        Ok(Some(Game {
            black_name,
            white_name,
            black_elo,
            white_elo,
            outcome,
            timectl_inc,
            timectl_sec,
            moves: self.moves.clone(),
        }))
    }

    fn header(&mut self, key: &[u8], val: pgn_reader::RawHeader<'_>) {
        let value = val.decode_utf8_lossy().to_string();
        match key {
            b"White" => self.headers.white_name = Some(value.into()),
            b"Black" => self.headers.black_name = Some(value.into()),
            b"WhiteElo" => self.headers.white_elo = Some(value.parse().unwrap()),
            b"BlackElo" => self.headers.black_elo = Some(value.parse().unwrap()),
            b"TimeControl" => {
                if value != "-" {
                    let mut timectl = value.split('+');
                    let sec = timectl.next().unwrap_or("0");
                    let inc = timectl.next().unwrap_or("0");

                    self.headers.timectl_sec = Some(sec.parse().unwrap());
                    self.headers.timectl_inc = Some(inc.parse().unwrap());
                } else {
                    self.headers.timectl_sec = Some(i32::MAX);
                    self.headers.timectl_inc = Some(i32::MAX);
                }
            }
            b"Result" => match value.as_str() {
                "1-0" => self.headers.outcome = Some(Outcome::WhiteWin),
                "0-1" => self.headers.outcome = Some(Outcome::BlackWin),
                "1/2-1/2" => self.headers.outcome = Some(Outcome::Draw),
                _ => self.skip = true,
            },
            b"Termination" => {
                if value != "Normal" {
                    self.skip = true;
                }
            }
            _ => {}
        }
    }

    fn end_headers(&mut self) -> Skip {
        Skip(self.skip)
    }

    fn begin_variation(&mut self) -> Skip {
        Skip(true)
    }

    fn san(&mut self, san_plus: SanPlus) {
        let mv = san_plus.san.to_move(&self.board).unwrap();
        let board = self.board.clone();
        self.moves
            .push(Move::new(mv.from().unwrap(), mv.to(), mv.promotion()));
        self.board = board.play(&mv).unwrap();
    }
}

#[test]
fn test_visitor() {
    let mut visitor = PgnVisitor::new();
    let mut reader = pgn_reader::BufferedReader::new_cursor(include_str!("testfiles/single.pgn"));
    let game = reader
        .read_game(&mut visitor)
        .unwrap()
        .unwrap()
        .unwrap()
        .unwrap();
    assert_eq!(game.white_name, "Dominguez Perez, Leinier");
    assert_eq!(game.black_name, "Navara, David");
    assert_eq!(game.white_elo, 2739);
    assert_eq!(game.black_elo, 2737);
    assert_eq!(game.outcome, Outcome::WhiteWin);
    assert_eq!(game.timectl_sec, 600);
    assert_eq!(game.timectl_inc, 0);
    assert_eq!(
        game.moves
            .into_iter()
            .map(|m| m.bitfield)
            .collect::<Vec<_>>(),
        &[
            2147, 1429, 1627, 1948, 2276, 1364, 1748, 981, 2667, 2469, 1058, 1819, 2217, 3028, 537,
            3558, 66, 2460, 1635, 1988, 587, 1355, 139, 267, 1561, 713, 1616, 587, 2072, 1810,
            2258, 1746, 8, 708, 2572, 1502, 798, 2526, 2644, 276, 1569, 1299, 2153, 908, 3114, 388,
            3608, 1958, 1058, 779, 2189, 4047, 861, 2471, 1563, 1217, 2674, 72, 1886, 2543, 2740,
            544, 3258, 3063, 1966, 3583, 1758, 1015, 3004, 4029, 3893, 2082, 3186
        ]
    );
}
