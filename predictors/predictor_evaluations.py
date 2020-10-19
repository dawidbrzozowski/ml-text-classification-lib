from typing import List

from predictors.predictor import Predictor
from predictors.presets import PRESETS
import numpy as np


def deep_predictor_test_on_sample(predictor: Predictor, text: str, desired_label=1) -> List[tuple]:
    print(f'Probabilities on text: {text}')
    whole_sentence_prob = predictor.predict(text)[0][desired_label]
    words = text.split()
    print(whole_sentence_prob)
    probs_without_word = {}
    for word in words:
        subtext = text.replace(word, '')
        print(f'Probabilities on text: {subtext}. \n Word ommited: {word}')
        probs_without_word[word] = predictor.predict(subtext)[0][desired_label]
        print(probs_without_word[word])
    std_deviation = np.std([probs_without_word[word] for word in words])
    impact_on_text = {word: (probs_without_word[word] - whole_sentence_prob)/std_deviation for word in words}
    return sorted(impact_on_text.items(), key=lambda item: item[1])


if __name__ == '__main__':
    preset_name = 'glove_rnn_predictor'
    preset = PRESETS[preset_name]
    predictor = Predictor(preset)
    print(deep_predictor_test_on_sample(predictor, "Me as fuck the past few weeks or more. I love you all I’m just dealin"))