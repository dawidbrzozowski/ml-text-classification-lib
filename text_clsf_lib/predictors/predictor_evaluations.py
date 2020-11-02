from typing import List

from text_clsf_lib.predictors.predictor import Predictor
import numpy as np


def deep_predictor_test_on_sample(predictor: Predictor, text: str, desired_label=1) -> List[tuple]:
    """
    This function should be used for analyzing predictions made by predictor.
    The function will print the importance of each word when making a prediction.
    :param predictor: Predictor object.
    :param text: Text to be analyzed.
    :param desired_label: Label that you are interested in.
     For example, if you want to analyze why is the text offensive and model output is [NotOffensive, Offensive],
     then the desired_label should be equal to 1.
    :return: Words from the text sorted by their impact on the prediction.
    In case of offensive language classification, this would be words from the most offensive to the least one.
    """
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
