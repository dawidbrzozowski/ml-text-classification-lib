import re
from abc import abstractmethod
from typing import List, Tuple
import en_core_web_sm

from utils.files_io import load_json

NER_CONVERTER_DEF_PATH = 'preprocessing/cleaning/resources/ner_converter.json'


class DataCleaner:
    """
    This class is meant to be responsible for cleaning text data and output data.
    It should remove damaged records and process the rest.
    TODO It should be split on two classes just like in vectorizer, where each one is responsible for text and output.
    """

    @abstractmethod
    def clean(self, data: List[dict]) -> Tuple[list, list]:
        pass


class BaselineDataCleaner(DataCleaner):
    def clean(self, data: List[dict]):
        return data


class TextCleaner:
    def __init__(self,
                 replace_numbers=False,
                 use_ner=False,
                 use_ner_converter=False,
                 use_twitter_data_preprocessing=False):
        self.replace_numbers = replace_numbers
        self.ner_tagger = en_core_web_sm.load() if use_ner else None
        self.ner_converter = None
        self.use_twitter_data_preprocessing = use_twitter_data_preprocessing
        if use_ner and use_ner_converter:
            self.ner_converter = load_json(NER_CONVERTER_DEF_PATH)

    def clean(self, texts: List[str]):
        print('Started data preprocessing...')
        if self.use_twitter_data_preprocessing:
            print('Twitter data preprocessing...')
            texts = self._preprocess_twitter_data(texts)
        if self.ner_tagger:
            print('NER tagging...')
            texts = self._perform_ner_on_texts(texts)
        if self.replace_numbers:
            print('Replacing numbers...')
            texts = self._replace_numbers(texts)
        print('Data preprocessing finished!')
        return texts

    def _perform_ner_on_texts(self, texts):
        processed_texts = []
        for text in texts:
            ents = self.ner_tagger(text).ents
            for ent in ents:
                convert = lambda label: label if self.ner_converter is None else self.ner_converter[label]
                text = text.replace(str(ent), convert(ent.label_))
            processed_texts.append(text)
        return processed_texts

    def _replace_numbers(self, texts):
        number_pattern = re.compile(r'[-+]?[.\d]*[\d]+[:,.\d]*')
        return [number_pattern.sub('<number>', text) for text in texts]

    def _preprocess_twitter_data(self, texts):
        """ Inspired from https://nlp.stanford.edu/projects/glove/preprocess-twitter.rb"""
        # Different regex parts for smiley faces
        processed_texts = []
        for text in texts:
            eyes = "[8:=;]"
            nose = "['`-]?"
            text = re.sub(r"https?://\S+\b|www\.(\w+\.)+\S*", "<url>", text)
            text = re.sub("@\w+", "<user>", text)
            text = re.sub(f"{eyes}{nose}[)d]+|[)d]+{nose}{eyes}", "<smile>", text, flags=re.IGNORECASE)
            text = re.sub(f'{eyes}{nose}p+', '<lolface>', text, flags=re.IGNORECASE)
            text = re.sub(f'{eyes}{nose}\(+|\)+{nose}{eyes}', '<sadface>', text)
            text = re.sub(f'{eyes}{nose}[\\\/|l*]', '<neutralface>', text)
            text = re.sub("/", " / ", text)
            text = re.sub('<3', '<heart>', text)
            text = re.sub(f'[-+]?[.\d]*[\d]+[:,.\d]*', '<number>', text)
            text = re.sub(f'#(?=\w+)', '<hashtag> ', text)
            processed_texts.append(text)
        return processed_texts


class OutputCleaner:
    """
    Output cleaner will remove rows, for which output is invalid.
    """

    def clean(self, data: List[dict]):
        correct_data = []
        for sample in data:
            if sample['offensive'] in (0, 1):
                correct_data.append(sample)
        return correct_data


class PresetDataCleaner(DataCleaner):
    def __init__(self, text_cleaner: TextCleaner, output_cleaner: OutputCleaner):
        self.text_cleaner = text_cleaner
        self.output_cleaner = output_cleaner

    def clean(self, data: List[dict]) -> Tuple[list, list]:
        texts = [sample['text'] for sample in data]
        cleaned_texts = self.text_cleaner.clean(texts)
        cleaned_data = []
        for sample, cleaned_text in zip(data, cleaned_texts):
            cleaned_data.append({
                'text': cleaned_text,
                'offensive': sample['offensive']})
        cleaned_data = self.output_cleaner.clean(cleaned_data)
        cleaned_texts = [sample['text'] for sample in cleaned_data]
        cleaned_outputs = [sample['offensive'] for sample in cleaned_data]
        return cleaned_texts, cleaned_outputs


if __name__ == '__main__':
    tw = TextCleaner(use_twitter_data_preprocessing=True, use_ner=True, use_ner_converter=True)
    print(tw.clean(['@USER claims that #NationalDay <3 CI/CD is not important to Donald Trump :-/']))
