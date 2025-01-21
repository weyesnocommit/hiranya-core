from multiprocessing import Queue
from typing import List, Tuple

from spacy.tokens import Token, Doc

from common.ml import MLDataPreprocessor, sampling_function
from common.nlp import Pos, CapitalizationMode
from config.nlp_config import CAPITALIZATION_COMPOUND_RULES, STRUCTURE_MODEL_TRAINING_MAX_SIZE, \
	STRUCTURE_MODEL_TEMPERATURE, MAX_SEQUENCE_LEN
from models.model_common import MLModelScheduler, MLModelWorker


import torch
import torch.nn as nn
import numpy as np
#from tensorflow.keras.preprocessing.sequence import pad_sequences

import torch
import torch.nn.functional as F

def pad_sequences(sequences, maxlen, padding='post', value=0):
	# Convert sequences to a list of tensors if they aren't already
	sequences = [torch.tensor(seq, dtype=torch.long) for seq in sequences]
	
	# Create a tensor of shape (num_sequences, maxlen) filled with the padding value
	padded_sequences = torch.full((len(sequences), maxlen), value, dtype=torch.long)
	
	for i, seq in enumerate(sequences):
		if padding == 'post':
			padded_sequences[i, :len(seq)] = seq
		elif padding == 'pre':
			padded_sequences[i, -len(seq):] = seq
		else:
			raise ValueError("Padding type {} not understood".format(padding))
	
	return padded_sequences

class StructurePreprocessor(MLDataPreprocessor):
	def __init__(self):
		MLDataPreprocessor.__init__(self, 'StructurePreprocessor')

	def get_preprocessed_data(self) -> Tuple:
		from keras.preprocessing.sequence import pad_sequences
		structure_data = pad_sequences(self.data, StructureModel.SEQUENCE_LENGTH, padding='post')
		structure_labels = np.array(self.labels)
		return structure_data, structure_labels

	def preprocess(self, doc: Doc) -> bool:
		#tekob from gpu ok
		if len(self.data) >= STRUCTURE_MODEL_TRAINING_MAX_SIZE:
			return False

		sequence = []
		previous_item = None
		for sentence_idx, sentence in enumerate(doc.sents):
			if len(self.data) >= STRUCTURE_MODEL_TRAINING_MAX_SIZE:
				return False

			for token_idx, token in enumerate(sentence):
				item = StructureFeatureAnalyzer.analyze(
					token, CapitalizationMode.from_token(token, CAPITALIZATION_COMPOUND_RULES))
				label = item

				if len(sequence) == 0:
					# Offset data by one, making label point to the next data item
					sequence.append(PoSCapitalizationMode(Pos.NONE, CapitalizationMode.NONE).to_embedding())
				else:
					sequence.append(previous_item)

				# We only want the latest SEQUENCE_LENGTH items
				sequence = sequence[-StructureModel.SEQUENCE_LENGTH:]

				self.data.append(sequence.copy())
				self.labels.append(label)

				previous_item = item

			# Handle EOS after each sentence
			item = PoSCapitalizationMode(Pos.EOS, CapitalizationMode.NONE).to_embedding()
			label = item

			sequence.append(previous_item)

			# We only want the latest SEQUENCE_LENGTH items
			sequence = sequence[-StructureModel.SEQUENCE_LENGTH:]

			self.data.append(sequence.copy())
			self.labels.append(label)

			previous_item = item
		return True



class PoSCapitalizationMode:
	def __init__(self, pos: Pos, mode: CapitalizationMode):
		self.pos = pos
		self.mode = mode

	def __repr__(self):
		return f"{self.pos.name}_{self.mode.name}"

	def to_embedding(self) -> int:
		return self.pos.value * len(CapitalizationMode) + self.mode.value

	@staticmethod
	def from_embedding(embedding: int):
		pos_part = int(embedding / len(CapitalizationMode))
		mode_part = int(embedding % len(CapitalizationMode))
		return PoSCapitalizationMode(Pos(pos_part), CapitalizationMode(mode_part))

class StructureFeatureAnalyzer:
	NUM_FEATURES = len(Pos) * len(CapitalizationMode)

	@staticmethod
	def analyze(token, mode: CapitalizationMode):
		pos = Pos.from_token(token)
		mode = PoSCapitalizationMode(pos, mode)
		return mode.to_embedding()

class PyTorchModel(nn.Module):
	def __init__(self, num_features, latent_dim, sequence_length):
		super(PyTorchModel, self).__init__()
		self.embedding = nn.Embedding(num_features, num_features)
		self.lstm = nn.LSTM(num_features, latent_dim, batch_first=True, dropout=0)
		self.dense = nn.Linear(latent_dim, num_features)
		self.softmax = nn.Softmax(dim=1)

	def forward(self, x):
		x = self.embedding(x)
		x, _ = self.lstm(x)
		x = self.dense(x[:, -1, :])  # Use the last output of the sequence
		x = self.softmax(x)
		return x

class StructureModel:
	SEQUENCE_LENGTH = MAX_SEQUENCE_LEN

	def __init__(self, use_gpu: bool = False):
		self.device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")
		latent_dim = StructureModel.SEQUENCE_LENGTH * 16
		self.model = PyTorchModel(
			num_features=StructureFeatureAnalyzer.NUM_FEATURES,
			latent_dim=latent_dim,
			sequence_length=StructureModel.SEQUENCE_LENGTH
		).to(self.device)
		self.pad_sequences = pad_sequences

	def train(self, data, labels, epochs=1, validation_split=0.2):
		val_size = int(len(data) * validation_split)
		indices = np.arange(len(data))
		np.random.shuffle(indices)
		val_indices = indices[:val_size]
		train_indices = indices[val_size:]

		train_data, val_data = data[train_indices], data[val_indices]
		train_labels, val_labels = labels[train_indices], labels[val_indices]

		train_data = torch.tensor(train_data, dtype=torch.long).to(self.device)
		train_labels = torch.tensor(train_labels, dtype=torch.long).to(self.device)
		val_data = torch.tensor(val_data, dtype=torch.long).to(self.device)
		val_labels = torch.tensor(val_labels, dtype=torch.long).to(self.device)

		optimizer = torch.optim.Adam(self.model.parameters())
		criterion = nn.CrossEntropyLoss()

		for epoch in range(epochs):
			self.model.train()
			optimizer.zero_grad()
			outputs = self.model(train_data)
			loss = criterion(outputs, train_labels)
			loss.backward()
			optimizer.step()

			print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item()}")

	def predict(self, num_sentences: int, sampling_config: dict) -> List[PoSCapitalizationMode]:
		self.model.eval()
		predictions = []
		sequence = [[0]]  # Start with a single token (e.g., start token)
		eos_count = 0  # Counter for end-of-sentence tokens
		idx = 0  # Iteration counter
		max_iterations = 200  # Increased max iterations for more flexibility
		temperature_increase_step = 0.1  # Step size for temperature adjustment
		max_temperature = 2.0  # Maximum temperature to avoid completely random outputs
		recent_tokens = []  # Track recent tokens to penalize repetition
		max_repeats = 3  # Maximum allowed repeats for a token
		min_unique_ratio = 0.3  # Minimum unique token ratio to detect low diversity

		while eos_count < num_sentences and idx < max_iterations:
			padded_sequence = self.pad_sequences(sequence, maxlen=StructureModel.SEQUENCE_LENGTH, padding='post')
			
			if not isinstance(padded_sequence, torch.Tensor):
				padded_sequence = torch.tensor(padded_sequence, dtype=torch.long, device=self.device)
			else:
				padded_sequence = padded_sequence.to(self.device)

			with torch.no_grad():
				prediction = self.model(padded_sequence).cpu().numpy()[0]

			for token in recent_tokens[-max_repeats:]:
				prediction[token] *= 0.5  # Reduce the probability of repeated tokens

			index = sampling_function(prediction, sampling_config)
			predictions.append(index)
			recent_tokens.append(index)

			if PoSCapitalizationMode.from_embedding(index).pos == Pos.EOS:
				eos_count += 1

			sequence[0].append(index)
			sequence[0] = sequence[0][-StructureModel.SEQUENCE_LENGTH:]  # Keep the sequence within the max length

			idx += 1
			if idx > 25 and eos_count == 0:
				print("[structure] XDD TOO MANY TIRES - Increasing temperature to encourage diversity")
				sampling_config['temperature'] = min(
					sampling_config['temperature'] + temperature_increase_step,
					max_temperature
				)

			if idx > 50 and eos_count == 0:
				print("[structure] Injecting randomness to break the loop")
				random_index = np.random.randint(0, len(sequence[0]))
				sequence[0][random_index] = np.random.randint(0, StructureFeatureAnalyzer.NUM_FEATURES)

			if idx > 75 and eos_count == 0:
				print("[structure] Falling back to greedy sampling to break the loop")
				sampling_config['temperature'] = 0.0  # Greedy sampling (always pick the most likely token)

			unique_tokens = set(predictions)
			unique_ratio = len(unique_tokens) / len(predictions)
			if unique_ratio < min_unique_ratio:
				print("[structure] Low diversity detected - Stopping early")
				break

			if idx >= max_iterations:
				print("[structure] Max iterations reached - Stopping early")
				break

		modes = [PoSCapitalizationMode.from_embedding(embedding) for embedding in predictions]
		return modes


	def load(self, path):
		self.model.load_state_dict(torch.load(path, map_location=self.device))
		self.model.to(self.device)

	def save(self, path):
		torch.save(self.model.state_dict(), path)

class StructureModelWorker(MLModelWorker):
	def __init__(self, read_queue: Queue, write_queue: Queue, use_gpu: bool = False):
		MLModelWorker.__init__(self, name='SentenceStructureModelWorker', read_queue=read_queue,
							   write_queue=write_queue,
							   use_gpu=use_gpu)

	def run(self):
		self._model = StructureModel(use_gpu=self._use_gpu)
		MLModelWorker.run(self)

	def predict(self, *data, sampling_config) -> List[PoSCapitalizationMode]:
		return self._model.predict(num_sentences=data[0][0], sampling_config=sampling_config)

	def train(self, *data):
		return self._model.train(data=data[0][0], labels=data[0][1], epochs=data[0][2])

	def save(self, *data):
		return self._model.save(path=data[0][0])

	def load(self, *data):
		return self._model.load(path=data[0][0])


class StructureModelScheduler(MLModelScheduler):
	def __init__(self, use_gpu: bool = False):
		MLModelScheduler.__init__(self)
		self._worker = StructureModelWorker(read_queue=self._write_queue, write_queue=self._read_queue,
											use_gpu=use_gpu)

	def predict(self, num_sentences: int, sampling_config: dict):
		return self._predict(num_sentences, sampling_config=sampling_config)

	def train(self, data, labels, epochs=1):
		return self._train(data, labels, epochs)

	def save(self, path):
		return self._save(path)

	def load(self, path):
		return self._load(path)
