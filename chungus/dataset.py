import numpy as np
import tensorflow as tf
import os

CZECHERNET_DATA_PREFIXES = ["2019_03", "2019_07", "2022_02", "2022_03"]


def get_train_data(datasets):
    files = list()
    for prefix in datasets:        
        if prefix not in CZECHERNET_DATA_PREFIXES:
            raise ValueError(f"Dataset prefix '{prefix}' does not exist (valid: {CZECHERNET_DATA_PREFIXES})")
        for file in os.listdir('data/train'):
            if file.startswith(prefix):
                files.append(file)
    for file in files:
        path = f"data/train/{file}"
        print(f"Loading training data from {path}")
        data = np.load(path)
        yield (data['x'], data['y'])
        

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
    data = np.load('data/test.npz')
    return tf.data.Dataset.from_tensor_slices((data['x'], data['y']))

def get_data(datasets, batch_size):
    train_dataset = tf.data.Dataset.from_generator(
        lambda: get_train_data(datasets),
        output_signature=(
            tf.TensorSpec(shape=(None, 8, 8, 13), dtype=tf.float32), 
            tf.TensorSpec(shape=(None, 3,), dtype=tf.float32)
        ),
    ).flat_map(lambda x, y: tf.data.Dataset.from_tensor_slices((x, y))).prefetch(tf.data.AUTOTUNE).batch(batch_size).shuffle(10_000)
    
    return train_dataset, get_test_data().batch(batch_size)