# Facial Emotion Recognition using CNN
# Import required libraries
import numpy as np
import pandas as pd
import cv2
import os
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, MaxPooling2D, Dropout, Flatten, Dense, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Set random seed for reproducibility
np.random.seed(42)

# Define paths
DATA_PATH = os.path.join(os.getcwd(), 'fer2013', 'fer2013.csv')
MODEL_PATH = 'emotion_model.h5'

# Load and preprocess data
def load_data():
    print("Loading data...")
    df = pd.read_csv(DATA_PATH)
    
    # Convert pixels to numpy arrays
    pixels = df['pixels'].apply(lambda x: np.array(x.split(' '), dtype='float32'))
    X = np.vstack(pixels.values)  # Convert list of arrays to single numpy array
    X = X.reshape(-1, 48, 48, 1)  # Reshape to (num_samples, 48, 48, 1)
    
    # Normalize pixel values to [-1, 1]
    X = (X - 127.5) / 127.5
    
    # Convert labels to one-hot encoding
    y = pd.get_dummies(df['emotion']).values
    
    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    return X_train, X_test, y_train, y_test

# Create the model
def create_model(input_shape=(48, 48, 1), num_classes=7):
    print("Creating model...")
    model = Sequential()
    
    # First Convolutional Block
    model.add(Conv2D(32, (3, 3), padding='same', input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    
    # Second Convolutional Block
    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    
    # Third Convolutional Block
    model.add(Conv2D(128, (3, 3), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    
    # Fully Connected Layer
    model.add(Flatten())
    model.add(Dense(512))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    
    # Output Layer
    model.add(Dense(num_classes, activation='softmax'))
    
    return model

def train_model():
    # Load data
    X_train, X_test, y_train, y_test = load_data()
    
    # Create model
    model = create_model()
    
    # Compile model
    model.compile(optimizer=Adam(learning_rate=0.001), 
                 loss='categorical_crossentropy', 
                 metrics=['accuracy'])
    
    # Callbacks
    callbacks = [
        EarlyStopping(patience=10, verbose=1, restore_best_weights=True),
        ReduceLROnPlateau(factor=0.1, patience=5, min_lr=0.00001, verbose=1),
        ModelCheckpoint(MODEL_PATH, save_best_only=True, verbose=1)
    ]
    
    # Data augmentation
    datagen = ImageDataGenerator(
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True
    )
    
    # Train model
    print("Training model...")
    history = model.fit(
        datagen.flow(X_train, y_train, batch_size=64),
        steps_per_epoch=len(X_train) // 64,
        epochs=50,
        validation_data=(X_test, y_test),
        callbacks=callbacks
    )
    
    return model, history

if __name__ == "__main__":
    print("Starting Facial Emotion Recognition Training...")
    model, history = train_model()
    print("Training completed! Model saved as:", MODEL_PATH)
