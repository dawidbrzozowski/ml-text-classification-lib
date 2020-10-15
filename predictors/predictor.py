from predictors.presets import PRESETS


class Predictor:
    def __init__(self, preset: dict):
        self.preprocessor = preset['preprocessor_func'](preset)
        self.model_runner = preset['model_func'](preset)

    def predict(self, text: list or str):
        preprocessed = self.preprocessor.preprocess([text]) if type(text) is str else self.preprocessor.preprocess(text)
        return self.model_runner.run(preprocessed).tolist()


if __name__ == '__main__':
    preset_name = 'glove_ff_predictor'
    preset = PRESETS[preset_name]
    inp = "Barack Obama"
    predictor = Predictor(preset)
    print(predictor.predict(inp))