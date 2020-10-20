from predictors.presets import PRESETS


class Predictor:
    def __init__(self, preset: dict):
        self.preprocessor = preset['preprocessor_func'](preset)
        self.model_runner = preset['model_func'](preset)

    def predict(self, text: list or str):
        preprocessed = self.preprocessor.clean_vectorize(text)
        return self.model_runner.run(preprocessed).tolist()


if __name__ == '__main__':
    preset_name = 'glove_rnn_predictor'
    preset = PRESETS[preset_name]
    inp = ["Me as fuck the past few weeks or more. I love you all I’m just dealin w a lot and trying the best I can❤️I’m the distant friend. The I haven’t texted u back and we havent spoken in 2 weeks but I still love u friend. 🥺🥺"]
    predictor = Predictor(preset)
    print(predictor.predict(inp))