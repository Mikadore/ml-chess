import chessers
import chess
import time, os
import numpy as np
from util import fmtsize
from enum import Enum
from multiprocessing import Pool


class Outcome(Enum):
    WhiteWin = 1
    Draw = 0
    BlackWin = -1
    
    def encode(self):
        match self:
            case Outcome.WhiteWin:
                return np.array([1,0,0])
            case Outcome.Draw:
                return np.array([0,1,0])
            case Outcome.BlackWin:
                return np.array([0,0,1])
    


def fromRustOutcome(outcome) -> Outcome:
    match outcome:
        case "1-0":
            return Outcome.WhiteWin
        case "0-1":
            return Outcome.BlackWin
        case "1/2-1/2":
            return Outcome.Draw
        case _:
            raise ValueError('invalid outcome')

def positions_encoded(moves):
    board = chess.Board()
    positions = list()
    
    for move in moves:
        board.push_uci(move)
        
        pos = np.zeros((8, 8, 13))
        for x in range(8):
            for y in range(8):
                piece = board.piece_at(chess.square(x, y))
                if piece != None:
                    if piece.color == chess.WHITE:
                        match piece.piece_type:
                            case chess.PAWN:
                                pos[x, y, 0] = 1
                            case chess.KNIGHT:
                                pos[x, y, 1] = 1
                            case chess.BISHOP:
                                pos[x, y, 2] = 1
                            case chess.ROOK:
                                pos[x, y, 3] = 1
                            case chess.QUEEN:
                                pos[x, y, 4] = 1
                            case chess.KING:
                                pos[x, y, 5] = 1
                    else:
                        match piece.piece_type:
                            case chess.PAWN:
                                pos[x, y, 6] = 1
                            case chess.KNIGHT:
                                pos[x, y, 7] = 1
                            case chess.BISHOP:
                                pos[x, y, 8] = 1
                            case chess.ROOK:
                                pos[x, y, 9] = 1
                            case chess.QUEEN:
                                pos[x, y, 10] = 1
                            case chess.KING:
                                pos[x, y, 11] = 1
                if board.turn == chess.WHITE:
                    pos[x, y, 12] = 1
                else:
                    pos[x, y, 12] = -1
        positions.append(pos)
    return positions

def to_moves(game):
    return game.moves()

def collect_positions(filepath):    
    size = os.stat(filepath).st_size
    start = time.time()
    pool = Pool(16)
    
        
    loader = chessers.pgn.GameLoader(filepath)
    loader = map(to_moves, loader)
    positions = pool.map(positions_encoded, loader)

    end = time.time()
    throughput = size/(end - start)
    print(f"Processed {len(positions)} positions ({fmtsize(throughput)}/s)")
    