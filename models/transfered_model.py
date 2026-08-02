from models.model import Model
from tensorflow.keras import Sequential, layers, models
from tensorflow.keras.layers.experimental.preprocessing import Rescaling
from tensorflow.keras.optimizers import RMSprop, Adam

class TransferedModel(Model):
    
    def _define_model(self, input_shape, categories_count):
        from tensorflow.keras.models import load_model

        base_model = load_model('results/your_model_name.keras')
        base_model.trainable = False

        x = base_model.layers[-2].output
        x = layers.Dense(32, activation='relu')(x)
        output = layers.Dense(categories_count, activation='softmax')(x)

        self.model = models.Model(inputs=base_model.input, outputs=output)
    
    def _compile_model(self):
        self.model.compile(
            optimizer=RMSprop(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy'],
        )
