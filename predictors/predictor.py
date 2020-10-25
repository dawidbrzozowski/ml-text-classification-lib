from predictors.presets import PRESETS


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
    preset_name = 'tfidf_predictor'
    preset = PRESETS[preset_name]
    inp = ["Me as fuck the past few weeks or more. I love you all I’m just dealin w a lot and trying the best I can❤️I’m the distant friend. The I haven’t texted u back and we havent spoken in 2 weeks but I still love u friend. 🥺🥺"]
    predictor = Predictor(preset)
    print(predictor.predict(inp))