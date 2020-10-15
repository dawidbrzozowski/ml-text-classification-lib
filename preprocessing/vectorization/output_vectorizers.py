from abc import abstractmethod
from typing import List
import numpy as np


class OutputVectorizer:
    @abstractmethod
    def fit(self, output: List[dict]):
        pass

    @abstractmethod
    def vectorize(self, output: List[dict]):
        pass


class BasicOutputVectorizer(OutputVectorizer):
    def __init__(self, threshold=0.5):
        self.threshold = threshold

    def fit(self, output: List[dict]):
        pass

    def vectorize(self, output: List[dict]):
        averages = [o['average'] for o in output]
        return np.array([1 if average >= self.threshold else 0 for average in averages])
