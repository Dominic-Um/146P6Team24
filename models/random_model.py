from models.model import Model
from tensorflow.keras import Sequential, layers, models
from tensorflow.keras.layers.experimental.preprocessing import Rescaling
from tensorflow.keras.optimizers import RMSprop, Adam
import numpy as np

class RandomModel(Model):
    def _define_model(self, input_shape, categories_count):
        # Load your base model 
        base_model = models.load_model("results/basic_model_30_epochs_timestamp_1785804752.keras")

        # Freeze all layers so training does NOT update them
        for layer in base_model.layers:
            layer.trainable = False

        # Randomize the weights of the loaded model
        self._randomize_layers(base_model)

        base_model_output = base_model.layers[-2].output
        x = layers.Dense(32, activation="relu", name='random_dense')(base_model_output)
        output = layers.Dense(categories_count, activation="softmax", name='random_output')(x)

        self.model = models.Model(inputs=base_model.input, outputs=output)
    
    def _compile_model(self):
        self.model.compile(
            optimizer=RMSprop(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy'],
        )

    @staticmethod
    def _randomize_layers(model):
        for layer in model.layers:
            weights = layer.get_weights()
            if len(weights) == 0:
                continue

            new_weights = []
            for w in weights:
                # Randomize with same shape
                new_w = np.random.standard_normal(w.shape) * 0.05
                new_weights.append(new_w)

            layer.set_weights(new_weights)
