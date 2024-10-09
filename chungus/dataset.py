import numpy as np
import tensorflow as tf
import os, time
import chessers
from pathlib import Path

def get_train_data():
    files = os.listdir('data/train/2022_02')
    files.sort()
    files = list(map(lambda f: Path(f'data/train/2022_02/{f}').resolve(), files))
    print(f"Using training data from {files}")
    loader = chessers.TrainDataLoader(files, 3)
    for td in loader:
        yield td.get_ins(), td.get_outs()
        

#def get_train_data(datasets, batch_size):
#    for data in get_train_data_npz(datasets):
#        x = data['x']
#        y = data['y']
#        num_samples = x.shape[0]
#        for start_idx in range(0, num_samples, batch_size):
#            end_idx = start_idx + batch_size
#            if end_idx < num_samples:
#                yield (x[start_idx:end_idx], y[start_idx:end_idx])

def get_test_data():
    data = chessers.TrainData.load('data/test.bin')
    return tf.data.Dataset.from_tensor_slices((data.get_ins(), data.get_outs()))

def get_data(batch_size):
    train_dataset = tf.data.Dataset.from_generator(
        lambda: get_train_data(),
        output_signature=(
            tf.TensorSpec(shape=(None, 8, 8, 37), dtype=tf.float32), 
            tf.TensorSpec(shape=(None, 3,), dtype=tf.float32)
        ),
    ).flat_map(lambda x, y: tf.data.Dataset.from_tensor_slices((x, y))).prefetch(tf.data.AUTOTUNE).shuffle(10_000).batch(batch_size, drop_remainder=True)
    
    return train_dataset, get_test_data().batch(batch_size)
