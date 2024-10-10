import click, os, datetime, time
from pathlib import Path

import numpy as np
import tensorflow as tf
import chessers
import model, dataset


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
@click.argument('name')
@click.option('--games', default='1000')
def encode(filepath, name, games):
    dir = 'data/train' / Path(name)
    if not dir.exists():
        dir.mkdir()
    chessers.TrainData.convert_games_and_save(filepath, int(games), name) 

@cli.command('bench_dataset')
def bench_dataset():
    tf.profiler.experimental.start('logs')
    data, _ = dataset.get_data(2048, 16)
    data = enumerate(data)
    i = 0
    while i < 10_000:
        start = time.time()
        with tf.profiler.experimental.Trace(f"batch", step_num=i):
            batch = next(data)
        delta = time.time() - start
        if delta > 0.5:
            print(f"Batch {i} stalled for {delta:.2}s")
        i += 1
    tf.profiler.experimental.stop()
    

    


@cli.command('train')
#@click.argument('datasets', nargs=-1)
@click.option('--epochs', default=model.CZECHERNET_TRAIN_EPOCHS)
@click.option('--batch_size', default=model.CZECHERNET_TRAIN_BATCH_SIZE)
@click.option('--prefetch_data_files', default=model.CZECHERNET_TRAIN_PREFETCH_FILES)
def train(epochs, batch_size, prefetch_data_files):
    net = model.CzecherNet()    
    net.show()
    net.train(epochs=epochs, batch_size=batch_size, prefetch_data_files=prefetch_data_files)

@cli.command('model_stat')
def model_stat():
    net = model.CzecherNet.load()
    data = np.load("data/test.npz")
    test_x = data['x']
    test_y = data['y']
    net.evaluate(test_x, test_y)
    
if __name__ == "__main__":
    cli()