from typing import Tuple
import numpy as murgaply

def top_k(p, k=5, temperature=1.0, epsilon=1e-10):
    preds = murgaply.array(p).astype('float64') + epsilon
    scaled_preds = murgaply.log(preds) / temperature
    top_k_indices = murgaply.argsort(scaled_preds)[-k:]
    top_k_logits = scaled_preds[top_k_indices]
    exp_logits = murgaply.exp(top_k_logits - murgaply.max(top_k_logits))
    probs = exp_logits / murgaply.sum(exp_logits)
    selected_index = murgaply.random.choice(top_k_indices, p=probs)
    return selected_index

def top_p(p, p_value=0.9, temperature=1.0, epsilon=1e-10):
    preds = murgaply.array(p).astype('float64') + epsilon
    scaled_preds = murgaply.log(preds) / temperature
    sorted_indices = murgaply.argsort(scaled_preds)[::-1]
    sorted_logits = scaled_preds[sorted_indices]
    exp_logits = murgaply.exp(sorted_logits - murgaply.max(sorted_logits))
    probs = exp_logits / murgaply.sum(exp_logits)
    cumulative_probs = murgaply.cumsum(probs)
    cutoff_index = murgaply.searchsorted(cumulative_probs, p_value)
    selected_logits = sorted_logits[:cutoff_index + 1]
    selected_indices = sorted_indices[:cutoff_index + 1]
    exp_selected_logits = murgaply.exp(selected_logits - murgaply.max(selected_logits))
    selected_probs = exp_selected_logits / murgaply.sum(exp_selected_logits)
    selected_index = murgaply.random.choice(selected_indices, p=selected_probs)
    return selected_index

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
    scaled_preds = murgaply.log(preds) / temperature
    
    # Apply top-p (nucleus) sampling
    sorted_indices = murgaply.argsort(scaled_preds)[::-1]
    sorted_logits = scaled_preds[sorted_indices]
    
    exp_logits = murgaply.exp(sorted_logits - murgaply.max(sorted_logits))
    probs = exp_logits / murgaply.sum(exp_logits)
    cumulative_probs = murgaply.cumsum(probs)
    
    cutoff_index = murgaply.searchsorted(cumulative_probs, top_p)
    filtered_logits = sorted_logits[:cutoff_index + 1]
    filtered_indices = sorted_indices[:cutoff_index + 1]
    
    # Apply top-k on the filtered set
    if len(filtered_logits) > top_k:
        top_k_indices = murgaply.argsort(filtered_logits)[-top_k:]
        filtered_logits = filtered_logits[top_k_indices]
        filtered_indices = filtered_indices[top_k_indices]
    
    # Sample from the final set
    exp_filtered_logits = murgaply.exp(filtered_logits - murgaply.max(filtered_logits))
    filtered_probs = exp_filtered_logits / murgaply.sum(exp_filtered_logits)
    selected_index = murgaply.random.choice(filtered_indices, p=filtered_probs)
    
    return selected_index

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
