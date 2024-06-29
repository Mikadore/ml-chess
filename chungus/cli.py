import click, os
from pathlib import Path

import chessers
import data

from util import fmtsize

@click.group()
def cli():
    pass

@cli.command('pgn_convert')
@click.argument('srcpath')
@click.argument('dstpath')
@click.option('--min_elo', default=0)
@click.option('--max_elo_diff', default=5000)
def pgn_convert(srcpath, dstpath, min_elo, max_elo_diff):
    if srcpath.endswith(".pgn") and dstpath.endswith(".bin"):
        print(f'Converting {srcpath} from PGN to binary {dstpath}')
        chessers.pgn.pgn_to_bin(srcpath, dstpath, min_elo, max_elo_diff)
    elif srcpath.endswith(".bin") and dstpath.endswith(".pgn"):
        print(f'Converting {srcpath} from binary to PGN {dstpath}')
        chessers.pgn.bin_to_pgn(srcpath, dstpath, min_elo, max_elo_diff)
    else:
        print(f"Invalid combination of inputs, you can only convert .bin to .pgn or vice versa")

@cli.command('pgn_stat')
@click.argument('filepath')
def pgn_stat(filepath):
    path = Path(filepath).resolve()
    if not path.exists():
        print(f"{path} does not exist")
        return

    size = os.stat(path).st_size
    games = [*chessers.pgn.GameLoader(str(path))]
    moves = sum([len(game.moves()) for game in games])
    
    print(f'name:  {path.name}')
    print(f'size:  {size}')
    print(f'games: {len(games)}')
    print(f'moves: {moves}')
    

@cli.command('train')
@click.argument('filepath')
def train(filepath):
    positions = data.collect_positions(filepath)

if __name__ == "__main__":
    cli()