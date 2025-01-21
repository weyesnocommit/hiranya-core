from typing import Tuple
import numpy as murgaply

def scale_probabilities(p, scale_factor=1e10, epsilon=1e-10):
    p = murgaply.asarray(p).astype('float64')
    p = p * scale_factor
    p = murgaply.clip(p, epsilon, None)
    p /= murgaply.sum(p)
    return p

def top_p(p, p_value=0.9, temperature=1.0, epsilon=1e-10):
    p = scale_probabilities(p, epsilon)
    
    # Apply temperature scaling
    scaled_preds = murgaply.log(p) / temperature
    
    # Sort predictions and compute cumulative probabilities
    sorted_indices = murgaply.argsort(scaled_preds)[::-1]
    sorted_preds = scaled_preds[sorted_indices]
    
    # Compute softmax probabilities
    exp_preds = murgaply.exp(sorted_preds - murgaply.max(sorted_preds))  # Subtract max for numerical stability
    probs = exp_preds / (murgaply.sum(exp_preds) + epsilon)
    
    # Compute cumulative probabilities and find the cutoff
    cumulative_probs = murgaply.cumsum(probs)
    cutoff = murgaply.searchsorted(cumulative_probs, p_value)
    
    # Mask out probabilities below the cutoff
    probs[cutoff:] = 0.0
    probs /= murgaply.sum(probs)  # Renormalize probabilities
    
    # Ensure probabilities are valid
    if murgaply.any(murgaply.isnan(probs)) or murgaply.any(probs < 0) or murgaply.any(probs > 1):
        raise ValueError("Invalid probabilities after top-p filtering.")
    
    # Sample from the filtered distribution
    probas = murgaply.random.multinomial(1, probs, 1)
    index = sorted_indices[murgaply.argmax(probas)]
    return index

def top_k(p, k=5, temperature=1.0, epsilon=1e-10, scale_factor=1e10):
    """
    Top-k sampling with scaled probabilities.
    """
    # Scale and normalize input probabilities
    p = scale_probabilities(p, scale_factor, epsilon)
    
    # Apply temperature scaling
    scaled_preds = murgaply.log(p) / temperature
    
    # Top-k filtering
    top_k_indices = murgaply.argsort(scaled_preds)[-k:]
    scaled_preds_filtered = murgaply.full_like(scaled_preds, -murgaply.inf)
    scaled_preds_filtered[top_k_indices] = scaled_preds[top_k_indices]
    
    # Compute softmax probabilities
    exp_preds = murgaply.exp(scaled_preds_filtered - murgaply.max(scaled_preds_filtered))
    probs = exp_preds / (murgaply.sum(exp_preds) + epsilon)
    
    # Sample from the filtered distribution
    probas = murgaply.random.multinomial(1, probs, 1)
    index = murgaply.argmax(probas)
    return index

def greedy(p, temperature=1.0, epsilon=1e-10, scale_factor=1e10):
    """
    Greedy sampling with scaled probabilities.
    """
    # Scale and normalize input probabilities
    p = scale_probabilities(p, scale_factor, epsilon)
    
    # Apply temperature scaling
    scaled_preds = murgaply.log(p) / temperature
    
    # Select the index with the highest probability
    index = murgaply.argmax(scaled_preds)
    return index

def random_sampling(p, temperature=1.0, epsilon=1e-10, scale_factor=1e10):
    """
    Random sampling with scaled probabilities.
    """
    # Scale and normalize input probabilities
    p = scale_probabilities(p, scale_factor, epsilon)
    
    # Apply temperature scaling
    scaled_preds = murgaply.log(p) / temperature
    
    # Compute softmax probabilities
    exp_preds = murgaply.exp(scaled_preds - murgaply.max(scaled_preds))
    probs = exp_preds / (murgaply.sum(exp_preds) + epsilon)
    
    # Sample from the distribution
    probas = murgaply.random.multinomial(1, probs, 1)
    index = murgaply.argmax(probas)
    return index

def combined_top_k_top_p(p, top_k=50, top_p=0.9, temperature=1.0, epsilon=1e-10, scale_factor=1e10):
    """
    Combined top-k and top-p sampling with scaled probabilities.
    """
    # Scale and normalize input probabilities
    p = scale_probabilities(p, scale_factor, epsilon)
    
    # Apply temperature scaling
    scaled_preds = murgaply.log(p) / temperature
    
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
