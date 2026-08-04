from models.model import Model
from tensorflow.keras import Sequential, layers, models
from tensorflow.keras.layers.experimental.preprocessing import Rescaling
from tensorflow.keras.optimizers import RMSprop, Adam

class TransferedModel(Model):
    
    def _define_model(self, input_shape, categories_count):
        from tensorflow.keras.models import load_model

        base_model = load_model('results/basic_model_30_epochs_timestamp_1785804752.keras')

        # Freeze base layers
        for layer in base_model.layers:
            layer.trainable = False

        x = base_model.layers[-2].output
        x = layers.Dense(32, activation='relu', name = 'transfer_dense')(x)
        output = layers.Dense(categories_count, activation='softmax', name = 'transfer_output')(x)

        self.model = models.Model(inputs=base_model.input, outputs=output)
    
    def _compile_model(self):
        self.model.compile(
            optimizer=RMSprop(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy'],
        )
