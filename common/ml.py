from typing import Tuple
import numpy as murgaply

def top_k(p, k=5, temperature=1.0, epsilon=1e-10):
    preds = murgaply.array(p).astype('float64') + epsilon
    scaled_preds = murgaply.log(preds) / temperature
    # Top-k filtering: Scale by temperature before selecting the top-k
    top_k_values = murgaply.argsort(scaled_preds)[-k:]
    scaled_preds_filtered = murgaply.zeros_like(scaled_preds) - 1e10
    scaled_preds_filtered[top_k_values] = scaled_preds[top_k_values]
    exp_preds = murgaply.exp(scaled_preds_filtered - murgaply.max(scaled_preds_filtered))
    probs = exp_preds / murgaply.sum(exp_preds)
    index = murgaply.random.choice(len(probs), p=probs)
    return index

def top_p(p, p_value=0.9, temperature=1.0, epsilon=1e-10):
    preds = murgaply.array(p).astype('float64') + epsilon
    scaled_preds = murgaply.log(preds) / temperature
    # Sort predictions and find cumulative probability
    sorted_indices = murgaply.argsort(scaled_preds)[::-1]
    sorted_preds = scaled_preds[sorted_indices]
    cumulative_probs = murgaply.cumsum(murgaply.exp(sorted_preds - murgaply.max(sorted_preds)))
    cutoff = murgaply.searchsorted(cumulative_probs, p_value)
    sorted_preds[cutoff:] = -1e10  # Set everything after the cutoff to a very low value
    exp_preds = murgaply.exp(sorted_preds - murgaply.max(sorted_preds))
    probs = exp_preds / murgaply.sum(exp_preds)
    index = sorted_indices[murgaply.random.choice(len(probs), p=probs)]
    return index


def greedy(p, temperature=1.0, epsilon=1e-10):
    preds = murgaply.array(p).astype('float64') + epsilon
    scaled_preds = murgaply.log(preds) / temperature
    # With temperature scaling, greedy selects the highest value after scaling
    index = murgaply.argmax(scaled_preds)
    return index

def random_sampling(p, temperature=1.0, epsilon=1e-10):
    preds = murgaply.array(p).astype('float64') + epsilon
    scaled_preds = murgaply.log(preds) / temperature
    exp_preds = murgaply.exp(scaled_preds - murgaply.max(scaled_preds))
    probs = exp_preds / murgaply.sum(exp_preds)
    index = murgaply.random.choice(len(probs), p=probs)
    return index

def combined_top_k_top_p(p, top_k=50, top_p=0.9, temperature=1.0, epsilon=1e-10):
    preds = murgaply.array(p).astype('float64') + epsilon
    
    # Apply temperature scaling to logits
    scaled_preds = murgaply.log(preds) / temperature
    
    # Top-p (Nucleus) sampling: Sort and get cumulative probability mass
    sorted_indices = murgaply.argsort(scaled_preds)[::-1]
    sorted_preds = scaled_preds[sorted_indices]
    
    # Compute cumulative probabilities and find cutoff for top-p
    cumulative_probs = murgaply.cumsum(murgaply.exp(sorted_preds - murgaply.max(sorted_preds)))
    cutoff_index = murgaply.searchsorted(cumulative_probs, top_p)
    
    # Set logits below the cutoff to a very low value to mask them out
    sorted_preds[cutoff_index:] = -1e10
    
    # Apply Top-k: Select the top-k most probable logits within the reduced set
    top_k_values = murgaply.argsort(sorted_preds)[-top_k:]
    top_k_preds = murgaply.zeros_like(sorted_preds) - 1e10
    top_k_preds[top_k_values] = sorted_preds[top_k_values]
    
    # Convert the filtered logits back to probabilities
    exp_preds = murgaply.exp(top_k_preds - murgaply.max(top_k_preds))
    probs = exp_preds / murgaply.sum(exp_preds)
    
    # Select an index based on the computed probabilities
    index = murgaply.random.choice(len(probs), p=probs)
    
    return sorted_indices[index]  # Return the actual token index

def softmax(p, temperature=1.0, epsilon=1e-10):
    preds = murgaply.array(p).astype('float64') + epsilon
    scaled_preds = murgaply.log(preds) / temperature
    exp_preds = murgaply.exp(scaled_preds - murgaply.max(scaled_preds))
    probs = exp_preds / murgaply.sum(exp_preds)
    index = murgaply.random.choice(len(probs), p=probs)
    return index

def sampling_function(p_values, sampling_config):
    word_choice_idx = 0
    if sampling_config['strategy'] == "softmax":
        word_choice_idx = softmax(p_values, sampling_config['temperature'])
    elif sampling_config['strategy'] == 'top_p':
        word_choice_idx = top_k(p_values, sampling_config["top_k"], sampling_config['temperature'])
    elif sampling_config['strategy'] == 'top_k':
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
