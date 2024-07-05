import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers, losses, optimizers, metrics

CZECHERNET_FILTERS = 256

CZECHERNET_TRAIN_BATCH_SIZE = 128
CZECHERNET_TRAIN_EPOCHS = 50

def create_model():
    input  = layers.Input(shape=(8, 8, 13))
    # 8x8x13 -> 6x6xFILT
    x = layers.Conv2D(CZECHERNET_FILTERS, (3, 3), activation='relu')(input)
    # 6x6xFILT -> 4x4xFILT
    x = layers.Conv2D(CZECHERNET_FILTERS, (3, 3), activation='relu')(x)
    # 4x4xFILT -> FILT
    x = layers.GlobalAveragePooling2D()(x)
    # Complete the net with some dense layers
    x = layers.Dense(CZECHERNET_FILTERS, activation='relu')(x)
    x = layers.Dense(CZECHERNET_FILTERS, activation='relu')(x)
    x = layers.Dense(3, activation='softmax')(x)
    
    model = keras.Model(inputs=input, outputs=x, name="CzecherNet")
    model.compile(loss=losses.CategoricalCrossentropy(), optimizer=optimizers.Adam(), metrics=[metrics.CategoricalCrossentropy()])
    return model

class CzecherNet:
    def __init__(self):
        self.model = create_model()
    
    def train(self, x, y):
        train_x = x[:-10_000]
        train_y = y[:-10_000]
        test_x  = x[-10_000:]
        test_y  = y[-10_000:]

        self.model.fit(train_x, train_y, batch_size=CZECHERNET_TRAIN_BATCH_SIZE, epochs=CZECHERNET_TRAIN_EPOCHS)
        self.evaluate(test_x, test_y)
        self.save()
    
    def evaluate(self, x, y):
        print(f"Evaluating model performance:")
        self.show() 
        loss, acc = self.model.evaluate(x, y, verbose=2)
        print(f"Loss = {loss:.2f} Accuracy = {acc:.2f}")
        return loss, acc
    
    def save(self):
        self.model.save_weights("data/checkpoint.ckpt")
        print(f"saved file to data/checkpoint.ckpt")
    
    def show(self):
        self.model.summary()
    
    def load() -> "CzecherNet":
        net = CzecherNet()
        net.model.load_weights("data/checkpoint.ckpt")
        print(f"Loaded data/checkpoint.ckpt")
