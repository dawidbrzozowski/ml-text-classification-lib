from django.apps import AppConfig

import sys
sys.path.append('/Users/dawidbrzozowski/Projects/offensive-language-semeval')

from text_clsf_lib.predictors.predictor import Predictor
from text_clsf_lib.predictors.presets import create_predictor_preset


class PostsConfig(AppConfig):
    preset = create_predictor_preset(
        model_name='glove_rnn',
        type_='embedding',
        max_seq_len=200,
        twitter_preprocessing=True)
    name = 'posts'
    predictor = Predictor(preset)
