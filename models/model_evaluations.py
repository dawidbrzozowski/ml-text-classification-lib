from typing import List
import numpy as np

def deep_model_test(
        texts: List[str],
        predictions: List[np.array],
        true_labels: List[int],
        show_all: bool = True,
        show_worst_samples: int or None = None,
        show_best_samples: int or None = None,
        show_wrong_samples_only: bool = False,
        to_file: None or str = None):

    if show_all:
        for text, prediction, true_label in zip(texts, predictions, true_labels):
            print(f'Processed text: {text}')
            print(f'Prediction: {prediction}')
            print(f'True label: {true_label}')
            print(f'------------------------')

    if show_worst_samples is not None:
        worst_samples = []  # always contains the worst sample in the last position. The rest is not in any order.
        for text, prediction, true_label in zip(texts, predictions, true_labels):
            pred_label = np.argmax(prediction)
            if pred_label != true_label:
                if len(worst_samples) < show_worst_samples:
                    worst_samples.append({
                        'text': text,
                        'prediction': prediction,
                        'true_label': true_label})
                    if len(worst_samples) == show_worst_samples:
                        worst_samples = sorted(worst_samples, key=lambda s: s['prediction'][s['true_label']])
                else:
                    if prediction[true_label] < worst_samples[-1]['prediction'][true_label]:
                        worst_samples.pop(-1)
                        i = 0
                        for i, sample in enumerate(worst_samples):
                            if prediction[true_label] < sample['prediction'][sample['true_label']]:
                                break

                        worst_samples.insert(i, {
                            'text': text,
                            'prediction': prediction,
                            'true_label': true_label})

        if len(worst_samples) != show_worst_samples:
            worst_samples = sorted(worst_samples, key=lambda s: s['prediction'][s['true_label']])

        for sample in worst_samples:
            print(f'Processed text: {sample["text"]}')
            print(f'Prediction: {sample["prediction"]}')
            print(f'True label: {sample["true_label"]}')
            print(f'------------------------')