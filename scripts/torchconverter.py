import tensorflow as tf
import torch
import torch.nn as nn
import numpy as np
from typing import List
import time

# Constants (replace with your actual values)
NUM_FEATURES = 180  # Updated to match the weights in the .h5 file
LATENT_DIM = 256    # Updated to match the weights in the .h5 file
SEQUENCE_LENGTH = 50# Replace with StructureModel.SEQUENCE_LENGTH

# Step 1: Recreate TensorFlow Model Architecture
def create_tensorflow_model(num_features, latent_dim, sequence_length):
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(num_features, num_features),
        tf.keras.layers.LSTM(latent_dim, dropout=0.2, return_sequences=False),
        tf.keras.layers.Dense(num_features, activation='softmax')
    ])
    model.build(input_shape=(None, sequence_length))
    return model

# Step 2: Load Weights into TensorFlow Model
def load_tensorflow_weights(tf_model, weights_path):
    tf_model.load_weights(weights_path)
    return tf_model

# Step 3: Extract Weights from TensorFlow Model
def extract_tensorflow_weights(tf_model):
    embedding_weights = tf_model.layers[0].get_weights()[0]  # Embedding layer weights
    lstm_weights = tf_model.layers[1].get_weights()          # LSTM layer weights
    dense_weights = tf_model.layers[2].get_weights()         # Dense layer weights
    return embedding_weights, lstm_weights, dense_weights

# Step 4: Define PyTorch Model
class PyTorchModel(nn.Module):
    def __init__(self, num_features, latent_dim, sequence_length):
        super(PyTorchModel, self).__init__()
        self.embedding = nn.Embedding(num_features, num_features)
        self.lstm = nn.LSTM(num_features, latent_dim, batch_first=True, dropout=0.2)
        self.dense = nn.Linear(latent_dim, num_features)
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, x):
        x = self.embedding(x)
        x, _ = self.lstm(x)
        x = self.dense(x[:, -1, :])  # Use the last output of the sequence
        x = self.softmax(x)
        return x

# Step 5: Load TensorFlow Weights into PyTorch Model
def load_weights_into_pytorch(pytorch_model, embedding_weights, lstm_weights, dense_weights):
    # Load embedding weights
    pytorch_model.embedding.weight.data = torch.tensor(embedding_weights, dtype=torch.float32)
    
    # Extract LSTM weights
    kernel, recurrent_kernel, bias = lstm_weights
    
    # Split the weights into individual gates
    W_i, W_f, W_c, W_o = np.split(kernel, 4, axis=1)
    U_i, U_f, U_c, U_o = np.split(recurrent_kernel, 4, axis=1)
    b_i, b_f, b_c, b_o = np.split(bias, 4)
    
    # Load weights into PyTorch LSTM
    with torch.no_grad():
        # Input-hidden weights
        pytorch_model.lstm.weight_ih_l0.data = torch.tensor(np.concatenate([W_i, W_f, W_c, W_o], axis=1).T, dtype=torch.float32)
        # Hidden-hidden weights
        pytorch_model.lstm.weight_hh_l0.data = torch.tensor(np.concatenate([U_i, U_f, U_c, U_o], axis=1).T, dtype=torch.float32)
        # Biases
        pytorch_model.lstm.bias_ih_l0.data = torch.tensor(np.concatenate([b_i, b_f, b_c, b_o]), dtype=torch.float32)
        pytorch_model.lstm.bias_hh_l0.data = torch.zeros_like(pytorch_model.lstm.bias_ih_l0.data)  # PyTorch uses separate biases
    
    # Load dense layer weights
    pytorch_model.dense.weight.data = torch.tensor(dense_weights[0].T, dtype=torch.float32)
    pytorch_model.dense.bias.data = torch.tensor(dense_weights[1], dtype=torch.float32)

# Step 6: Save PyTorch Model
def save_pytorch_model(pytorch_model, path):
    torch.save(pytorch_model.state_dict(), path)

# Step 7: Sampling Function (Replicated from TensorFlow Logic)
def sampling_function(prediction, sampling_config):
    temperature = sampling_config.get('temperature', 1.0)
    prediction = np.log(prediction) / temperature
    exp_preds = np.exp(prediction)
    prediction = exp_preds / np.sum(exp_preds)
    return np.random.choice(len(prediction), p=prediction)

# Step 8: Replicate TensorFlow Prediction Logic in TensorFlow
def predict_tensorflow(tf_model, num_sentences: int, sampling_config: dict) -> List[int]:
    predictions = []
    sequence = [[0]]
    eos_count = 0
    idx = 0

    while eos_count < num_sentences:
        # Pad the sequence
        padded_sequence = tf.keras.preprocessing.sequence.pad_sequences(sequence, maxlen=SEQUENCE_LENGTH, padding='post')
        
        # Predict
        prediction = tf_model.predict(padded_sequence, batch_size=1)[0]
        
        # Sample from the prediction
        index = sampling_function(prediction, sampling_config)
        
        # Check for EOS
        if index == 0:  # Replace with your EOS condition
            eos_count += 1
        
        # Append to predictions and update sequence
        predictions.append(index)
        sequence[0].append(index)
        sequence[0] = sequence[0][-SEQUENCE_LENGTH:]
        
        # Increment index and adjust temperature if necessary
        idx += 1
        if idx > 25:
            print("[structure] XDD TOO MANY TIRES")
            sampling_config['temperature'] += 0.1
        if idx > 50:
            break

    return predictions

# Step 9: Replicate TensorFlow Prediction Logic in PyTorch
def predict_pytorch(pytorch_model, num_sentences: int, sampling_config: dict) -> List[int]:
    predictions = []
    sequence = [[0]]
    eos_count = 0
    idx = 0

    while eos_count < num_sentences:
        # Pad the sequence
        padded_sequence = tf.keras.preprocessing.sequence.pad_sequences(sequence, maxlen=SEQUENCE_LENGTH, padding='post')
        padded_sequence = torch.tensor(padded_sequence, dtype=torch.long)
        
        # Predict
        with torch.no_grad():
            prediction = pytorch_model(padded_sequence).numpy()[0]
        
        # Sample from the prediction
        index = sampling_function(prediction, sampling_config)
        
        # Check for EOS
        if index == 0:  # Replace with your EOS condition
            eos_count += 1
        
        # Append to predictions and update sequence
        predictions.append(index)
        sequence[0].append(index)
        sequence[0] = sequence[0][-SEQUENCE_LENGTH:]
        
        # Increment index and adjust temperature if necessary
        idx += 1
        if idx > 25:
            print("[structure] XDD TOO MANY TIRES")
            sampling_config['temperature'] += 0.1
        if idx > 50:
            break

    return predictions

# Step 10: Compare Inference Times
def compare_inference_times(tf_model, pytorch_model, num_tests=10):
    # Generate a random input sequence
    input_data = np.random.randint(0, NUM_FEATURES, size=(1, SEQUENCE_LENGTH))
    
    # TensorFlow inference time
    tf_times = []
    for _ in range(num_tests):
        start_time = time.time()
        tf_model.predict(input_data, batch_size=1)
        tf_times.append(time.time() - start_time)
    
    # PyTorch inference time
    pytorch_times = []
    pytorch_input = torch.tensor(input_data, dtype=torch.long)
    for _ in range(num_tests):
        start_time = time.time()
        with torch.no_grad():
            pytorch_model(pytorch_input)
        pytorch_times.append(time.time() - start_time)
    
    # Print results
    print(f"TensorFlow Average Inference Time: {np.mean(tf_times):.6f} seconds")
    print(f"PyTorch Average Inference Time: {np.mean(pytorch_times):.6f} seconds")
    print(f"PyTorch is {np.mean(tf_times) / np.mean(pytorch_times):.2f}x faster than TensorFlow")

def verify_conversion(tf_model, pytorch_model, num_tests=5):
    for i in range(num_tests):
        input_data = np.random.randint(0, NUM_FEATURES, size=(1, SEQUENCE_LENGTH))  # Example input
        
        # TensorFlow output
        tf_output = tf_model.predict(input_data)
        
        # PyTorch output
        pytorch_input = torch.tensor(input_data, dtype=torch.long)
        pytorch_output = pytorch_model(pytorch_input).detach().numpy()
        
        # Print outputs
        print(f"Test {i + 1}:")
        print("TensorFlow Output:", tf_output)
        print("PyTorch Output:", pytorch_output)
        print("Outputs are close:", np.allclose(tf_output, pytorch_output, atol=1e-5))
        print()
import random
random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)
torch.manual_seed(42)

# Step 7: Deterministic Sampling Function
def deterministic_sampling(prediction):
    return np.argmax(prediction)  # Always select the index with the highest probability

# Step 8: Replicate TensorFlow Prediction Logic in TensorFlow (Deterministic)
def predict_tensorflow_deterministic(tf_model, num_sentences: int) -> List[int]:
    predictions = []
    sequence = [[0]]
    eos_count = 0
    idx = 0

    while eos_count < num_sentences:
        # Pad the sequence
        padded_sequence = tf.keras.preprocessing.sequence.pad_sequences(sequence, maxlen=SEQUENCE_LENGTH, padding='post')
        
        # Predict
        prediction = tf_model.predict(padded_sequence, batch_size=1)[0]
        
        # Sample from the prediction (deterministic)
        index = deterministic_sampling(prediction)
        
        # Check for EOS
        if index == 0:  # Replace with your EOS condition
            eos_count += 1
        
        # Append to predictions and update sequence
        predictions.append(index)
        sequence[0].append(index)
        sequence[0] = sequence[0][-SEQUENCE_LENGTH:]
        
        # Increment index
        idx += 1
        if idx > 50:
            break

    return predictions

# Step 9: Replicate TensorFlow Prediction Logic in PyTorch (Deterministic)
def predict_pytorch_deterministic(pytorch_model, num_sentences: int) -> List[int]:
    predictions = []
    sequence = [[0]]
    eos_count = 0
    idx = 0

    while eos_count < num_sentences:
        # Pad the sequence
        padded_sequence = tf.keras.preprocessing.sequence.pad_sequences(sequence, maxlen=SEQUENCE_LENGTH, padding='post')
        padded_sequence = torch.tensor(padded_sequence, dtype=torch.long)
        
        # Predict
        with torch.no_grad():
            prediction = pytorch_model(padded_sequence).numpy()[0]
        
        # Sample from the prediction (deterministic)
        index = deterministic_sampling(prediction)
        
        # Check for EOS
        if index == 0:  # Replace with your EOS condition
            eos_count += 1
        
        # Append to predictions and update sequence
        predictions.append(index)
        sequence[0].append(index)
        sequence[0] = sequence[0][-SEQUENCE_LENGTH:]
        
        # Increment index
        idx += 1
        if idx > 50:
            break

    return predictions

# Main Script
if __name__ == "__main__":
    # Step 1: Recreate TensorFlow model
    tf_model = create_tensorflow_model(NUM_FEATURES, LATENT_DIM, SEQUENCE_LENGTH)
    
    # Step 2: Load weights into TensorFlow model
    tf_model = load_tensorflow_weights(tf_model, "structure-model.weights.h5")
    
    # Step 3: Extract weights from TensorFlow model
    embedding_weights, lstm_weights, dense_weights = extract_tensorflow_weights(tf_model)
    
    # Step 4: Initialize PyTorch model
    pytorch_model = PyTorchModel(NUM_FEATURES, LATENT_DIM, SEQUENCE_LENGTH)
    
    # Step 5: Load TensorFlow weights into PyTorch model
    load_weights_into_pytorch(pytorch_model, embedding_weights, lstm_weights, dense_weights)
    
    # Step 6: Save PyTorch model
    save_pytorch_model(pytorch_model, "pytorch_model.pth")
    
    # Step 7: Verify conversion over multiple iterations
    verify_conversion(tf_model, pytorch_model, num_tests=5)
    
    # Step 8: Compare inference times
    compare_inference_times(tf_model, pytorch_model, num_tests=10)
    
    # Step 9: Test deterministic prediction logic for both models
    tf_predictions = predict_tensorflow_deterministic(tf_model, num_sentences=3)
    print("TensorFlow Predictions (Deterministic):", tf_predictions)
    
    pytorch_predictions = predict_pytorch_deterministic(pytorch_model, num_sentences=3)
    print("PyTorch Predictions (Deterministic):", pytorch_predictions)