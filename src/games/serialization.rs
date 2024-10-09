use super::game::*;
use eyre::{ensure, Result};
use std::fs::{File, OpenOptions};
use std::io::{BufReader, BufWriter, ErrorKind, Read, Write};
use std::path::Path;

const MAGIC: &[u8] = b"PGNSUX";

pub struct Encoder<W: Write> {
    inner: W,
    written: usize,
}

impl Encoder<BufWriter<File>> {
    pub fn open(p: &Path) -> Result<Self> {
        let f = OpenOptions::new()
            .write(true)
            .truncate(true)
            .create(true)
            .open(p)?;
        let w = BufWriter::new(f);
        Self::start(w)
    }
}

impl<W: Write> Encoder<W> {
    pub fn start(mut w: W) -> Result<Self> {
        w.write_all(MAGIC)?;
        Ok(Self {
            inner: w,
            written: MAGIC.len(),
        })
    }

    pub fn write_game(&mut self, game: &Game) -> Result<()> {
        let encoded = bincode::serialize(&game)?;
        self.inner.write_all(&encoded.len().to_le_bytes())?;
        self.inner.write_all(&encoded)?;
        self.written += encoded.len() + std::mem::size_of::<usize>();
        Ok(())
    }

    #[allow(unused)]
    pub fn bytes_written(&self) -> usize {
        self.written
    }
}

impl<W: Write> Drop for Encoder<W> {
    fn drop(&mut self) {
        self.inner.flush().unwrap();
    }
}

type DynReader = Box<dyn Read + Send>;

pub struct Decoder {
    inner: DynReader,
}

impl Decoder {
    pub fn open(p: &Path) -> Result<Decoder> {
        let f = OpenOptions::new().read(true).open(p)?;
        let r = BufReader::new(f);
        Decoder::start(Box::new(r))
    }
}

impl Decoder {
    pub fn start(mut r: DynReader) -> Result<Self> {
        let mut buf = [0; MAGIC.len()];
        r.read_exact(&mut buf)?;
        ensure!(buf == MAGIC, "File format corrupted");
        Ok(Self { inner: r })
    }

    pub fn read_game_raw(&mut self) -> Result<Option<Vec<u8>>> {
        let mut lenbuf = [0; std::mem::size_of::<usize>()];
        if let Err(e) = self.inner.read_exact(&mut lenbuf) {
            if e.kind() == ErrorKind::UnexpectedEof {
                return Ok(None);
            } else {
                return Err(e.into());
            }
        }

        let game_len = usize::from_le_bytes(lenbuf);

        let mut gamebuf = vec![0; game_len];
        self.inner.read_exact(&mut gamebuf)?;
        Ok(Some(gamebuf))
    }

    pub fn read_game(&mut self) -> Result<Option<Game>> {
        match self.read_game_raw() {
            Ok(Some(gamebuf)) => Ok(Some(bincode::deserialize(&gamebuf)?)),
            Ok(None) => Ok(None),
            Err(e) => Err(e),
        }
    }

    pub fn raw_iter(&mut self) -> RawGameIter<'_> {
        RawGameIter { inner: self }
    }
}

pub struct RawGameIter<'a> {
    inner: &'a mut Decoder
}

impl<'a> Iterator for RawGameIter<'a> {
    type Item = Vec<u8>;
    fn next(&mut self) -> Option<Self::Item> {
        self.inner.read_game_raw().unwrap()
    }
}

impl Iterator for Decoder {
    type Item = Game;
    fn next(&mut self) -> Option<Self::Item> {
        self.read_game().unwrap()
    }
}

#[cfg(test)]
pub fn decode_bin(r: DynReader) -> Result<Vec<Game>> {
    let mut dec = Decoder::start(r)?;
    let mut games = Vec::new();
    while let Some(game) = dec.read_game()? {
        games.push(game);
    }
    Ok(games)
}

#[test]
fn test_identical_decoding() {
    use std::io::Cursor;

    let test_bin = include_bytes!("testfiles/test.bin");
    let test_bin = Box::new(Cursor::new(test_bin));

    let mut bin_games = vec![];
    let mut decoder = Decoder::start(test_bin).unwrap();
    while let Some(g) = decoder.read_game().unwrap() {
        bin_games.push(g)
    }

    let test_pgn = include_bytes!("testfiles/test.pgn");
    let test_pgn = Cursor::new(test_pgn);

    let mut pgn_games = vec![];
    let mut visitor = super::pgn::PgnVisitor::new();
    let mut reader = pgn_reader::BufferedReader::new(test_pgn);
    while let Some(game) = reader.read_game(&mut visitor).unwrap() {
        if let Some(g) = game.unwrap() {
            pgn_games.push(g)
        }
    }

    for (a, b) in bin_games.iter().zip(pgn_games.iter()) {
        assert_eq!(a, b);
    }
}

#[test]
fn test_identity() {
    use std::io::Cursor;
    let pgn = include_bytes!("testfiles/single.pgn");

    let mut buf = vec![0; pgn.len()];
    let mut cursor = Cursor::new(&mut buf);
    let mut original_games = Vec::new();

    let mut encoder = Encoder::start(&mut cursor).unwrap();
    let mut visitor = super::pgn::PgnVisitor::new();
    let mut reader = pgn_reader::BufferedReader::new_cursor(pgn);
    while let Some(game) = reader.read_game(&mut visitor).unwrap() {
        if let Some(g) = game.unwrap() {
            encoder.write_game(&g).unwrap();
            original_games.push(g);
        }
    }

    let len = encoder.bytes_written();
    drop(encoder);

    let cursor = Cursor::new((&buf[..len]).to_vec());
    let mut decoder = Decoder::start(Box::new(cursor)).unwrap();
    let mut games = original_games.into_iter();
    while let Some(g) = decoder.read_game().unwrap() {
        assert_eq!(g, games.next().unwrap());
    }
}

#[test]
fn test_visitor() {
    use std::io::Cursor;
    let cursor = Box::new(Cursor::new(include_bytes!("testfiles/single.bin")));
    let games = decode_bin(cursor).unwrap();
    assert_eq!(games.len(), 1);

    let game = games[0].clone();
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
