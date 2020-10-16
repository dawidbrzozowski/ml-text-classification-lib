from predictors.presets import PRESETS


class Predictor:
    def __init__(self, preset: dict):
        self.preprocessor = preset['preprocessor_func'](preset)
        self.model_runner = preset['model_func'](preset)

    def predict(self, text: list or str):
        preprocessed = self.preprocessor.preprocess(text)['text_vectorized']
        return self.model_runner.run(preprocessed).tolist()


if __name__ == '__main__':
    preset_name = 'tfidf_predictor'
    preset = PRESETS[preset_name]
    inp = ["Netflix decides to finally release the second season of Mindhunter and it's on Flare-On 6 day?  You insidious bastards."]
    predictor = Predictor(preset)
    print(predictor.predict(inp))