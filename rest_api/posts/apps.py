from django.apps import AppConfig
import sys
import os

BASE_PATH = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(BASE_PATH)
from text_clsf_lib.predictors.predictor import Predictor
from text_clsf_lib.predictors.presets import create_predictor_preset
from semeval_utils.predictor import SemEvalPredictor

MODEL_SERVED = 'bpe_best_model'


class PostsConfig(AppConfig):
    preset = create_predictor_preset(model_name=MODEL_SERVED)
    name = 'posts'
    predictor = SemEvalPredictor(Predictor(preset))
