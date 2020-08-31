import os
from abc import abstractmethod

import numpy as np
from keras import Model
from keras.models import load_model, save_model

from configs.model_config_preparer import ModelPreparer
from configs.preprocessing_config_preparer import PreprocessingPreparer
from utils.files_io import load_json


class NNModelTrainer:
    def __init__(self, model_config, vectorization_meta_inf):
        model_preparer = ModelPreparer(model_config, vectorization_meta_inf)
        self.training_config = model_preparer.training_config
        self.model: Model = model_preparer.model

    def train(self, train_data):
        X_train, y_train = train_data
        self.model.fit(x=X_train,
                       y=y_train,
                       batch_size=self.training_config.batch_size,
                       epochs=self.training_config.epochs,
                       validation_split=self.training_config.validation_split,
                       callbacks=self.training_config.callbacks)

    def evaluate(self, test_data):
        X_test, y_test = test_data
        self.model.evaluate(x=X_test,
                            y=y_test)

    def save(self, save_dir, model_name):
        os.makedirs(save_dir, exist_ok=True)
        save_model(self.model, f'{save_dir}/{model_name}.h5')


class NNModelRunner:
    def __init__(self, model=None, model_path=None):
        self.model = model if model is not None else self.load_model(model_path)

    def load_model(self, model_path):
        return load_model(model_path)

    def evaluate(self, test_data):
        X_test, y_test = test_data
        self.model.evaluate(x=X_test,
                            y=y_test)

    @abstractmethod
    def run(self, data: list or np.array):
        pass

#
preprocessing_config = load_json('configs/data/preprocessing_config.json')
model_config = load_json('configs/data/model_config.json')
data_train = load_json('data/unprocessed/100000/train_corpus.json')
data_test = load_json('data/unprocessed/100000/test_corpus.json')
object_retr = PreprocessingPreparer(preprocessing_config)
data_preprocessor = object_retr.get_preprocessor()
data_preprocessor.fit(data_train)
data_train = data_preprocessor.preprocess(data_train)
data_test = data_preprocessor.preprocess(data_test)
vec_metainf = data_preprocessor.get_vectorization_metainf()
m_trainer = NNModelTrainer(model_config, vec_metainf)
m_trainer.train(data_train)
m_trainer.save('models/_models', 'first_model')

# runner = NNModelRunner(model_path='models/_models/first_model.h5')
# runner.evaluate(data_test)