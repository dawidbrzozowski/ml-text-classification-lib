from abc import abstractmethod
from typing import List


class DataCleaner:
    """
    This class is meant to be responsible for cleaning text data and output data.
    It should remove damaged records and process the rest.
    TODO It should be split on two classes just like in vectorizer, where each one is responsible for text and output.
    """

    @abstractmethod
    def fit(self, data: List[dict]):
        pass

    @abstractmethod
    def clean(self, data: List[dict]):
        pass


class BaselineDataCleaner(DataCleaner):
    def fit(self, data: List[dict]):
        return data

    def clean(self, data: List[dict]):
        return data
