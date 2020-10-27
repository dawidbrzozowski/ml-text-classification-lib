class Predictor:
    def __init__(self, preset: dict):
        preprocessing_params = preset['preprocessing_params']
        model_path = preset['model_path']
        self.preprocessor = preset['preprocessor_func'](preprocessing_params)
        self.model_runner = preset['model_func'](model_path)

    def predict(self, text: list or str):
        preprocessed = self.preprocessor.clean_vectorize(text)
        return self.model_runner.run(preprocessed).tolist()
