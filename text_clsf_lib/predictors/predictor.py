from text_clsf_lib.predictors.presets import create_predictor_preset


class Predictor:
    def __init__(self, preset: dict):
        preprocessing_params = preset['preprocessing_params']
        model_path = preset['model_path']
        self.preprocessor = preset['preprocessor_func'](preprocessing_params)
        self.model_runner = preset['model_func'](model_path)

    def predict(self, text: list or str):
        preprocessed = self.preprocessor.clean_vectorize(text)
        return self.model_runner.run(preprocessed).tolist()

if __name__ == '__main__':
    preset = create_predictor_preset(model_name='my_tfidf',
                                     type_='tfidf')
    pr = Predictor(preset)
    print(pr.predict('easy text with no special meaning fuck fucking trump'))