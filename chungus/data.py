import chessers
import time, os, subprocess
import numpy as np
from util import fmtsize

def collect_positions(filepath, threads):    
    size = os.stat(filepath).st_size
    start = time.time()
    num_cores = int(subprocess.run("nproc", check=True,shell=True,capture_output=True).stdout.strip())
    positions, outcomes = chessers.data.load_encoded(filepath, num_cores if not threads else int(threads))
    end = time.time()
    delta = end - start
    throughput = size/delta
    print(f"Processed {len(positions)} ({fmtsize(size)}b) positions in {delta:.2}s ({fmtsize(throughput)}b/s)")
    