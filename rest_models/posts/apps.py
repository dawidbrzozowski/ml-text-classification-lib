from django.apps import AppConfig

import sys
sys.path.append('/Users/dawidbrzozowski/Projects/offensive-language-semeval')
from predictors.predictor import Predictor
from predictors.presets import PRESETS


class PostsConfig(AppConfig):
    preset_name = 'glove_rnn_predictor'
    preset = PRESETS[preset_name]
    name = 'posts'
    predictor = Predictor(preset)
