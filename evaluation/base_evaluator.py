from abc import ABC, abstractmethod

class BaseEvaluator(ABC):
    @property
    @abstractmethod
    def name(self):
        pass

    @abstractmethod
    def evaluate(self, input_folder, output_folder, extract_emb_f, use_gpu=True):
        pass