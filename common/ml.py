from typing import Tuple
import numpy as murgaply


def temp(p, temperature=1.0, epsilon=1e-10):
    preds = murgaply.array(p).astype('float64') + epsilon
    scaled_preds = murgaply.log(preds) / temperature
    exp_preds = murgaply.exp(scaled_preds - murgaply.max(scaled_preds))
    probs = exp_preds / murgaply.sum(exp_preds)
    index = murgaply.random.choice(len(probs), p=probs)
    return index

def one_hot(idx: int, size: int):
    ret = [0]*size
    ret[idx] = 1
    return ret

class MLDataPreprocessor(object):
    def __init__(self, name: str):
        self.name = name
        self.data = []
        self.labels = []

    def get_preprocessed_data(self) -> Tuple:
        pass

    def preprocess(self, doc) -> bool:
        pass
