from models.model import Model
from tensorflow.keras import Sequential, layers
from tensorflow.keras.layers.experimental.preprocessing import Rescaling
from tensorflow.keras.optimizers import RMSprop

class BasicModel(Model):
    # Current best accuracy: 69% on 15 epochs
    def _define_model(self, input_shape, categories_count):
        self.model = Sequential([
        Rescaling(1./255, input_shape=input_shape),
        layers.Conv2D(16, (4,4), activation='relu'),
        layers.MaxPooling2D((4,4)),
        layers.Dropout(0.1),

        layers.Conv2D(32, (4,4), activation='relu'),
        layers.MaxPooling2D((3,3)),
        layers.Dropout(0.1),

        layers.Conv2D(48, (4,4), activation='relu'),
        layers.MaxPooling2D((2,2)),
        layers.Dropout(0.1),

        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.1),
        layers.Dense(categories_count, activation='softmax'),
    ])

    """
    Initial Version below. 60% accuracy on 6 epochs in train.py
    ---------------------------------------------
    def _define_model(self, input_shape, categories_count):
        self.model = Sequential([
        Rescaling(1./255, input_shape=input_shape),
        layers.Conv2D(16, 3, activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(32, 3, activation='relu'),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(categories_count, activation='softmax'),
    ])
    """

    
    def _compile_model(self):
        self.model.compile(
            optimizer=RMSprop(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy'],
        )
