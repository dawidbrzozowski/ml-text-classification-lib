from text_clsf_lib.predictors.predictor import Predictor
from text_clsf_lib.predictors.presets import create_predictor_preset

if __name__ == '__main__':
    preset = create_predictor_preset(model_name='tfidf_feedforward',
                                     type_='tfidf',
                                     twitter_preprocessing=False)
    inp = ["Me as fuck the past few weeks or more. I love you all I’m just dealin w a lot and trying the best I can❤️I’m the distant friend. The I haven’t texted u back and we havent spoken in 2 weeks but I still love u friend. 🥺🥺"]
    predictor = Predictor(preset)
    print(predictor.predict(inp))