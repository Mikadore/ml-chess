import click, os, time
from pathlib import Path

import numpy as np
import chessers
import data, model


from util import fmtsize, num_cores

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
    

@cli.command('encode')
@click.argument('filepath')
@click.argument('savepath')
@click.option('--threads', default=None, required=False)
def encode(filepath, savepath, threads):
    size = os.stat(filepath).st_size
    start = time.time()
    positions, outcomes = data.load_positions(filepath, num_cores() if not threads else int(threads))
    end = time.time()
    delta = end - start
    throughput = size/delta
    print(f"Processed {len(positions)} ({fmtsize(positions.nbytes + outcomes.nbytes)}b) positions in {delta:.2f}s ({fmtsize(throughput)}b/s)")
    print(f"Input  shape: {positions.shape}")
    print(f"Output shape: {outcomes.shape}")
    np.savez_compressed(savepath, x=positions, y=outcomes)
    print(f"Saved training data to {savepath}")

@cli.command('train')
@click.option('--dataset', default='train')
def train(dataset):
    net = model.CzecherNet()    
    data = np.load(f"data/{dataset}.npz")
    train_x = data['x']
    train_y = data['y']

    
    print(f"Loaded training data (in: {train_x.shape}, out: {train_y.shape})")
    net.show()
    net.train(train_x, train_y)

@cli.command('model_stat')
def model_stat():
    net = model.CzecherNet.load()
    data = np.load("data/test.npz")
    test_x = data['x']
    test_y = data['y']
    net.evaluate(test_x, test_y)

    
if __name__ == "__main__":
    cli()