from typing import List


class TextPreprocessor:

    def preprocess(self, texts: List[str]):
        pass


class EmbeddingTextPreprocessor(TextPreprocessor):
    def __init__(self):
        pass

    def preprocess(self, text: List[str]):
        pass


class TfIdfTextPreprocessor(TextPreprocessor):
    def __init__(self):
        pass

    def preprocess(self, text: List[str]):
        pass


class BasicOutputPreprocessor:
    def __init__(self, threshold=0.5):
        self.threshold = threshold

    def preprocess(self, output: List[dict]):
        averages = [o['average'] for o in output]
        return [1 if average >= self.threshold else 0 for average in averages]


class SemEvalDataPreprocessor:
    def __init__(self, text_preprocessor: TextPreprocessor = None, output_preprocessor=BasicOutputPreprocessor()):
        self.text_preprocessor = text_preprocessor
        self.output_preprocessor = output_preprocessor

    def preprocess(self, data: List[dict]):
        texts = [d['text'] for d in data]
        outputs = [{'average': d['average'], 'std': d['std']} for d in data]
        processed_texts = self.text_preprocessor.preprocess(texts)
        processed_output = self.output_preprocessor.preprocess(outputs)
        return processed_texts, processed_output
