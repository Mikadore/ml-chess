import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers, losses, optimizers, metrics, callbacks, regularizers
import dataset, chessers

CZECHERNET_FILTERS = 256
CZECHERNET_TRAIN_BATCH_SIZE = 64
CZECHERNET_TRAIN_EPOCHS = 50
CZECHERNET_TRAIN_PREFETCH_FILES = 3

CZECHERNET_CHECKPOINT_FILE = "data/checkpoint.weights.h5"

def create_model():
    input  = layers.Input(shape=(8, 8, 37))
    # First set of conv layers
    x = layers.Conv2D(CZECHERNET_FILTERS//4, (3, 3), activation='relu', padding='same')(input)
    x = layers.Conv2D(CZECHERNET_FILTERS//4, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPool2D((2, 2))(x)
    # Second set of conv layers
    x = layers.Conv2D(CZECHERNET_FILTERS//2, (3, 3), activation='relu', padding='same')(x)
    x = layers.Conv2D(CZECHERNET_FILTERS//2, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPool2D((2, 2))(x)
    # Last conv layer
    x = layers.Conv2D(CZECHERNET_FILTERS, (3, 3), activation='relu', padding='same')(x)
    x = layers.Conv2D(CZECHERNET_FILTERS, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.GlobalAveragePooling2D()(x)
    # Add dropout 
    x = layers.Dropout(0.5)(x)
    # Complete the net with some dense layers
    x = layers.Dense(CZECHERNET_FILTERS*3, activation='relu')(x)
    x = layers.Dense(CZECHERNET_FILTERS*2, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(CZECHERNET_FILTERS, activation='relu')(x)
    x = layers.Dense(3, activation='softmax')(x)
    
    model = keras.Model(inputs=input, outputs=x, name="CzecherNet")
    model.compile(loss=losses.CategoricalCrossentropy(), optimizer=optimizers.SGD(), metrics=[metrics.CategoricalCrossentropy()])
    return model

class CzecherNet:
    def __init__(self):
        self.model = create_model()
    
    def train(self, **kwargs):
        epochs = kwargs.get("epochs", CZECHERNET_TRAIN_EPOCHS)
        batch_size = kwargs.get("batch_size", CZECHERNET_TRAIN_BATCH_SIZE)
        prefetch_data_files = kwargs.get("prefetch_data_files", CZECHERNET_TRAIN_PREFETCH_FILES)

        print(f"Epochs = {epochs}, Batch size = {batch_size}")
        
        reduce_lr = callbacks.ReduceLROnPlateau(monitor='loss', verbose=1)
        early_stopping = callbacks.EarlyStopping(monitor='loss', patience=10, restore_best_weights=True, verbose=1)
        checkpoint_cb = callbacks.ModelCheckpoint('best.keras', monitor='loss', save_freq='epoch', verbose=1)

        train, test = dataset.get_data(batch_size, prefetch_data_files)
        
        self.model.fit(
            train,
            epochs=epochs,
            #batch_size=batch_size,
            validation_data=test,
            callbacks=[
                early_stopping,
                checkpoint_cb,
                reduce_lr,
            ])
        
        self.save()
        #self.evaluate(test)
    
    def evaluate(self, testdata):
        print(f"Evaluating model performance:")
        self.show() 
        loss, acc = self.model.evaluate(testdata, verbose=2)
        print(f"Loss = {loss:.2f} Accuracy = {acc:.2f}")
        return loss, acc
    
    def save(self):
        self.model.save_weights(CZECHERNET_CHECKPOINT_FILE)
        print(f"saved weights file to {CZECHERNET_CHECKPOINT_FILE}")
    
    def show(self):
        self.model.summary()
    
    def load() -> "CzecherNet":
        net = CzecherNet()
        net.model.load_weights(CZECHERNET_CHECKPOINT_FILE)
        print(f"Loaded {CZECHERNET_CHECKPOINT_FILE}")
        return net
