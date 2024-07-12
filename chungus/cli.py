import click, os, datetime
from pathlib import Path

import numpy as np
import tensorflow as tf
import chessers
import positions, model, dataset


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
@click.option('--games', default='10000')
def encode(filepath, savepath, games):
    positions.encode(filepath, int(games), savepath)


@cli.command('train')
@click.argument('datasets', nargs=-1)
@click.option('--epochs', default=model.CZECHERNET_TRAIN_EPOCHS)
@click.option('--batch_size', default=model.CZECHERNET_TRAIN_BATCH_SIZE)
@click.option('--profile', is_flag=True, default=False)
def train(datasets, epochs, batch_size, profile):
    if profile:
        print("Profiling...")
        logdir = "logs/scalars/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        tf.summary.create_file_writer(logdir)
        tf.profiler.experimental.start(logdir)
    datasets = [str(dataset) for dataset in datasets]
    print(f"Training using {','.join(datasets)}")
    net = model.CzecherNet()    
    net.show()
    net.train(datasets, epochs=epochs, batch_size=batch_size)
    if profile:
        tf.profiler.experimental.stop()

@cli.command('model_stat')
def model_stat():
    net = model.CzecherNet.load()
    data = np.load("data/test.npz")
    test_x = data['x']
    test_y = data['y']
    net.evaluate(test_x, test_y)

@cli.command('data_stat')
@click.argument('datasets', nargs=-1)
def data_stat(datasets):
    train, test = dataset.get_data(datasets, 1)
    print(fmtsize(sum([nda[0].nbytes + nda[1].nbytes for nda in train.take(1).as_numpy_iterator()])))
    
if __name__ == "__main__":
    cli()