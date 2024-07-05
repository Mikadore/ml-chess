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

def get_train_data(datasets):
    for data in get_train_data_npz(datasets):
        x = data['x']
        y = data['y']
        for i in range(x.shape[0]):
            yield (x[i], y[i]) 
 
 


def get_test_data():
    data = np.load('data/test.npz')
    return tf.data.Dataset.from_tensor_slices((data['x'], data['y']))

def get_data(datasets, batch_size):
    dataset = tf.data.Dataset.from_generator(
        lambda: get_train_data(datasets),
        output_signature=(
            tf.TensorSpec(shape=(8, 8, 13), dtype=tf.float32), 
            tf.TensorSpec(shape=(3,), dtype=tf.float32)
        ),
    ).prefetch(tf.data.AUTOTUNE)
    return dataset.batch(batch_size), get_test_data().batch(batch_size)