import chessers
import time, os
import numpy as np
from util import fmtsize, num_cores

def iter_batches(loader, batch_size):
    while True:
        positions, outcomes = loader.load_positions(batch_size)
        if len(positions) == 0:
            return
        else:
            yield np.stack(positions), np.stack(outcomes)

def encode(filepath: str, games_per_file, output_prefix):
    loader = chessers.data.PositionLoader(filepath, num_cores())
    start = time.time()
    batch_start = start
    i = 0
    total_positions = 0
    total_memory = 0
    for positions, outcomes in iter_batches(loader, games_per_file):
        path = f"data/{output_prefix}.{i}.npz"
        np.savez_compressed(path, x=positions, y=outcomes)
        i += 1
        total_positions += len(positions)
        total_memory += positions.nbytes + outcomes.nbytes
        end = time.time()
        delta = end-batch_start
        throughput = len(positions)/(end - batch_start)
        print(f"Loaded {len(positions)} positions, outcomes in {delta:.2f}s ({fmtsize(throughput)} pos/s)")
        print(f"Saved training data to {path}")
        batch_start = end
   
    delta = time.time() - start
    throughput = total_memory / delta 
    print(f"Processed {total_positions} ({fmtsize(total_memory)}b) positions in {delta:.2f}s ({fmtsize(throughput)}b/s)")