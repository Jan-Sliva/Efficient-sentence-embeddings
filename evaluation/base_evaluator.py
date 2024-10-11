from abc import ABC, abstractmethod
from architectures.base_retrieval_model import BaseRetrievalModel

class BaseEvaluator(ABC):
    @abstractmethod
    def __init__(self, **params):
        pass

    @property
    @abstractmethod
    def name(self):
        pass

    @abstractmethod
    def evaluate(self, retrieval_model : BaseRetrievalModel):
        pass
