import os
import numpy as np
from keras import Model
from keras.models import load_model, save_model

MODEL_SAVE_DIR = 'models/_models/'


class NNModelTrainer:
    def __init__(self, preset):
        model_builder = preset['model_builder_class'](preset)
        self.training_params = preset['training_params']
        self.model: Model = model_builder.prepare_model_architecture()

    def train(self, train_data, name='curr_model'):
        X_train, y_train = train_data
        self.model.fit(x=X_train,
                       y=y_train,
                       batch_size=self.training_params['batch_size'],
                       epochs=self.training_params['epochs'],
                       validation_split=self.training_params['validation_split'],
                       callbacks=self.training_params['callbacks'])
        self.save(MODEL_SAVE_DIR, name)

    def evaluate(self, test_data):
        X_test, y_test = test_data
        self.model.evaluate(x=X_test,
                            y=y_test)

    def save(self, save_dir, model_name):
        os.makedirs(save_dir, exist_ok=True)
        save_model(self.model, f'{save_dir}/{model_name}.h5')


class NNModelRunner:
    def __init__(self, model=None, model_path=None):
        self.model: Model = model if model is not None else load_model(model_path)

    def evaluate(self, test_data):
        X_test, y_test = test_data
        self.model.evaluate(x=X_test,
                            y=y_test)

    def run(self, data: list or np.array):
        return self.model.predict(data)
