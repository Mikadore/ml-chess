import numpy as np
import tensorflow as tf
import os

CZECHERNET_DATA_PREFIXES = ["2019_03", "2019_07", "2022_02", "2022_03"]


def get_train_data_npz(datasets):
    for prefix in datasets:        
        if prefix not in CZECHERNET_DATA_PREFIXES:
            raise ValueError(f"Dataset prefix '{prefix}' does not exist (valid: {CZECHERNET_DATA_PREFIXES})")
        for file in os.listdir('data/train'):
            if file.startswith(prefix):
                yield np.load(f"data/train/{file}")

def get_train_data(datasets, batch_size):
    for data in get_train_data_npz(datasets):
        x = data['x']
        y = data['y']
        num_samples = x.shape[0]
        for start_idx in range(0, num_samples, batch_size):
            end_idx = start_idx + batch_size
            if end_idx < num_samples:
                yield (x[start_idx:end_idx], y[start_idx:end_idx])

def get_test_data(batch_size):
    data = np.load('data/test.npz')
    x = data['x']
    y = data['y']
    num_samples = x.shape[0]
    for start_idx in range(0, num_samples, batch_size):
        end_idx = start_idx + batch_size
        if end_idx < num_samples:
            yield (x[start_idx:end_idx], y[start_idx:end_idx])

def get_data(datasets, batch_size):
    train_dataset = tf.data.Dataset.from_generator(
        lambda: get_train_data(datasets, batch_size),
        output_signature=(
            tf.TensorSpec(shape=(batch_size, 8, 8, 13), dtype=tf.float32), 
            tf.TensorSpec(shape=(batch_size, 3,), dtype=tf.float32)
        ),
    ).prefetch(tf.data.AUTOTUNE)
    test_dataset = tf.data.Dataset.from_generator(
        lambda: get_test_data(batch_size),
        output_signature=(
            tf.TensorSpec(shape=(batch_size, 8, 8, 13), dtype=tf.float32), 
            tf.TensorSpec(shape=(batch_size, 3,), dtype=tf.float32)
        ),
    )
    return train_dataset, test_dataset