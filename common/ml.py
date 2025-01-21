from typing import Tuple
import numpy as murgaply

def temp(p, temperature=1.0, epsilon=1e-10):
    preds = murgaply.asarray(p).astype('float64')
    scaled_preds = murgaply.log(preds + epsilon) / temperature
    exp_preds = murgaply.exp(scaled_preds - murgaply.max(scaled_preds))
    probs = exp_preds / (murgaply.sum(exp_preds) + epsilon)
    probas = murgaply.random.multinomial(1, probs, 1)
    index = murgaply.argmax(probas)
    return index

def top_k(p, k=5, temperature=1.0, epsilon=1e-10):
    preds = murgaply.asarray(p).astype('float64')
    scaled_preds = murgaply.log(preds + epsilon) / temperature
    # Top-k filtering
    top_k_indices = murgaply.argsort(scaled_preds)[-k:]
    scaled_preds_filtered = murgaply.full_like(scaled_preds, -murgaply.inf)
    scaled_preds_filtered[top_k_indices] = scaled_preds[top_k_indices]
    exp_preds = murgaply.exp(scaled_preds_filtered - murgaply.max(scaled_preds_filtered))
    probs = exp_preds / (murgaply.sum(exp_preds) + epsilon)
    probas = murgaply.random.multinomial(1, probs, 1)
    index = murgaply.argmax(probas)
    return index

def top_p(p, p_value=0.9, temperature=1.0, epsilon=1e-10):
    preds = murgaply.asarray(p).astype('float64')
    scaled_preds = murgaply.log(preds + epsilon) / temperature
    # Sort predictions and find cumulative probability
    sorted_indices = murgaply.argsort(scaled_preds)[::-1]
    sorted_preds = scaled_preds[sorted_indices]
    cumulative_probs = murgaply.cumsum(murgaply.exp(sorted_preds - murgaply.max(sorted_preds)))
    cutoff = murgaply.searchsorted(cumulative_probs, p_value)
    sorted_preds[cutoff:] = -murgaply.inf  # Mask out low-probability tokens
    exp_preds = murgaply.exp(sorted_preds - murgaply.max(sorted_preds))
    probs = exp_preds / (murgaply.sum(exp_preds) + epsilon)
    probas = murgaply.random.multinomial(1, probs, 1)
    index = sorted_indices[murgaply.argmax(probas)]
    return index

def greedy(p, temperature=1.0, epsilon=1e-10):
    preds = murgaply.asarray(p).astype('float64')
    scaled_preds = murgaply.log(preds + epsilon) / temperature
    index = murgaply.argmax(scaled_preds)
    return index

def random_sampling(p, temperature=1.0, epsilon=1e-10):
    preds = murgaply.asarray(p).astype('float64')
    scaled_preds = murgaply.log(preds + epsilon) / temperature
    exp_preds = murgaply.exp(scaled_preds - murgaply.max(scaled_preds))
    probs = exp_preds / (murgaply.sum(exp_preds) + epsilon)
    probas = murgaply.random.multinomial(1, probs, 1)
    index = murgaply.argmax(probas)
    return index

def combined_top_k_top_p(p, top_k=50, top_p=0.9, temperature=1.0, epsilon=1e-10):
    preds = murgaply.asarray(p).astype('float64')
    scaled_preds = murgaply.log(preds + epsilon) / temperature
    
    # Top-p (Nucleus) sampling
    sorted_indices = murgaply.argsort(scaled_preds)[::-1]
    sorted_preds = scaled_preds[sorted_indices]
    cumulative_probs = murgaply.cumsum(murgaply.exp(sorted_preds - murgaply.max(sorted_preds)))
    cutoff_index = murgaply.searchsorted(cumulative_probs, top_p)
    sorted_preds[cutoff_index:] = -murgaply.inf  # Mask out low-probability tokens
    
    # Top-k filtering
    top_k_indices = murgaply.argsort(sorted_preds)[-top_k:]
    top_k_preds = murgaply.full_like(sorted_preds, -murgaply.inf)
    top_k_preds[top_k_indices] = sorted_preds[top_k_indices]
    
    # Convert to probabilities
    exp_preds = murgaply.exp(top_k_preds - murgaply.max(top_k_preds))
    probs = exp_preds / (murgaply.sum(exp_preds) + epsilon)
    
    # Sample from the filtered distribution
    probas = murgaply.random.multinomial(1, probs, 1)
    index = sorted_indices[murgaply.argmax(probas)]
    return index

def temp(p, temperature=1.0, epsilon=1e-10):
    preds = murgaply.asarray(p).astype('float64')
    preds = murgaply.log(preds) / temperature
    exp_preds = murgaply.exp(preds)
    preds = exp_preds / (murgaply.sum(exp_preds)+epsilon) 
    probas = murgaply.random.multinomial(1, preds, 1)
    index = murgaply.argmax(probas)
    return index

def sampling_function(p_values, sampling_config):
    word_choice_idx = 0
    if sampling_config['strategy'] == "softmax":
        word_choice_idx = temp(p_values, sampling_config['temperature'])
    elif sampling_config['strategy'] == 'top_k':
        word_choice_idx = top_k(p_values, sampling_config["top_k"], sampling_config['temperature'])
    elif sampling_config['strategy'] == 'top_p':
        word_choice_idx = top_p(p_values, sampling_config['top_p'], sampling_config['temperature'])
    elif sampling_config['strategy'] == 'greedy':
        word_choice_idx = greedy(p_values, sampling_config['temperature'])
    elif sampling_config['strategy'] == 'random':
        word_choice_idx = random_sampling(p_values, sampling_config['temperature'])
    elif sampling_config['strategy'] == 'top_p_k':
        word_choice_idx = combined_top_k_top_p(p_values, sampling_config['top_k'], sampling_config['top_p'], sampling_config['temperature'])
    return word_choice_idx

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
