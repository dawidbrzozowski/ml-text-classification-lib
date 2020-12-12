from text_clsf_lib.predictors.predictor import Predictor
from math import ceil

from text_clsf_lib.predictors.presets import create_predictor_preset

OFFENSIVE_RATE_IDX = 1


class SemEvalPredictor:
    def __init__(self, predictor: Predictor):
        self.predictor = predictor

    def predict_for_text(self, text: str):
        prediction, cut_off_ratio = self.predictor.predict(text)
        curr_max_score = prediction[0][OFFENSIVE_RATE_IDX]
        cut_off_ratio = cut_off_ratio[0]
        if cut_off_ratio <= 1:
            return curr_max_score
        else:
            return max(curr_max_score,
                       max([self.predict_for_text(part) for part in split_text_n_parts(text, ceil(cut_off_ratio))]))


def split_text_n_parts(text, n:int):
    tokens = text.split(' ')
    rest = len(tokens) % n
    easy_divide_num_tokens = len(tokens) - rest
    step = len(tokens)//n
    parts = [tokens[i*step:(i+1)*step] for i in range(n)]
    parts[-1] += tokens[easy_divide_num_tokens: easy_divide_num_tokens+rest]
    return [' '.join(part) for part in parts]


if __name__ == '__main__':
    predictor = Predictor(create_predictor_preset('bpe_best_model'))
    semevalpred = SemEvalPredictor(predictor)
    text = 'Something definitely longer than 5 max seq len or even more FUCK YOU NIGGER'
    res = semevalpred.predict_for_text(text)
    print(res)