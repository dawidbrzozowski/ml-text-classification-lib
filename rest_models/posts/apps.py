from django.apps import AppConfig

import sys
sys.path.append('/Users/dawidbrzozowski/Projects/offensive-language-semeval')
from predictors import Predictor
from utils.files_io import load_json


class PostsConfig(AppConfig):
    name = 'posts'
    predictor = Predictor(load_json('configs/data/predictor_config.json'))
