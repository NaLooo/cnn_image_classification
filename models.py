from tensorflow.python.keras import Sequential
from tensorflow.python.keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Flatten
from keras.layers.normalization.batch_normalization import BatchNormalization
from tensorflow_addons.layers import AdaptiveAveragePooling2D
from torch import nn


class CNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 64, 3, 1, 1),
            nn.BatchNorm2d(num_features=64, affine=False),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.BatchNorm2d(num_features=64, affine=False),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 64*2, 3, 1, 1),
            nn.BatchNorm2d(num_features=64*2, affine=False),
            nn.ReLU(),
            nn.Conv2d(64*2, 64*2, 3, 1, 1),
            nn.BatchNorm2d(num_features=64*2, affine=False),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64*2, 64*4, 3, 1, 2),
            nn.BatchNorm2d(num_features=64*4, affine=False),
            nn.ReLU(),
            nn.Conv2d(64*4, 64*4, 3, 1, 1),
            nn.BatchNorm2d(num_features=64*4, affine=False),
            nn.ReLU(),
            nn.Conv2d(64*4, 64*4, 3, 1, 1),
            nn.BatchNorm2d(num_features=64*4, affine=False),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten(),
            nn.Linear(64*4,10),
            nn.Dropout(0.5),
            nn.Softmax(-1),
        )

    def forward(self, input):
        output = self.net(input)
        return output

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
