import chessers
import time, os, subprocess
import numpy as np
from util import fmtsize

def load_positions(filepath, threads):    
    positions, outcomes = chessers.data.load_positions(filepath, threads)
    positions = np.stack(positions)
    outcomes = np.stack(outcomes)
    return positions, outcomes