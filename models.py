from tensorflow.python.keras import Sequential
from tensorflow.python.keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Flatten
from keras.layers.normalization.batch_normalization import BatchNormalization
from tensorflow_addons.layers import AdaptiveAveragePooling2D


def vgg_16(input_shape=(28, 28, 1), categories=10):
    model = Sequential([
        Conv2D(32, 3, activation='relu', padding='same', input_shape=input_shape),
        Conv2D(64, 3, activation='relu', padding='same'),
        MaxPooling2D(2,2),
        BatchNormalization(),

        Conv2D(128, 3, activation='relu', padding='same'),
        Conv2D(128, 3, activation='relu', padding='same'),
        MaxPooling2D(2,2),
        BatchNormalization(),

        Conv2D(256, 3, activation='relu', padding='same'),
        Conv2D(256, 3, activation='relu', padding='same'),
        Conv2D(256, 3, activation='relu', padding='same'),
        MaxPooling2D(2, 2),
        BatchNormalization(),

        Conv2D(512, 3, activation='relu', padding='same'),
        Conv2D(512, 3, activation='relu', padding='same'),
        Conv2D(512, 3, activation='relu', padding='same'),
        MaxPooling2D(2, 1),
        BatchNormalization(),

        AdaptiveAveragePooling2D((1,1)),
        Flatten(),
        Dense(categories, activation='softmax'),
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model
