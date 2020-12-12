from django.apps import AppConfig

import sys
sys.path.append('/Users/dawidbrzozowski/Projects/offensive-language-semeval')

from text_clsf_lib.predictors.predictor import Predictor
from text_clsf_lib.predictors.presets import create_predictor_preset


class PostsConfig(AppConfig):
    preset = create_predictor_preset(model_name='bpe_best_model',
                                     type_='bpe')
    name = 'posts'
    predictor = Predictor(preset)
