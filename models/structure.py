from multiprocessing import Queue
from typing import List, Tuple

import numpy as np
from spacy.tokens import Token, Doc

from common.ml import MLDataPreprocessor, temp
from common.nlp import Pos, CapitalizationMode
from config.nlp_config import CAPITALIZATION_COMPOUND_RULES, STRUCTURE_MODEL_TRAINING_MAX_SIZE, \
	STRUCTURE_MODEL_TEMPERATURE, MAX_SEQUENCE_LEN
from models.model_common import MLModelScheduler, MLModelWorker


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


class PoSCapitalizationMode(object):
	def __init__(self, pos: Pos, mode: CapitalizationMode):
		self.pos = pos
		self.mode = mode

	def __repr__(self):
		return str(self.pos).split(".")[1] + "_" + str(self.mode).split(".")[1]

	def to_embedding(self) -> int:
		return self.pos.value * len(CapitalizationMode) + self.mode.value

	@staticmethod
	def from_embedding(embedding: int):
		pos_part = int(embedding / len(CapitalizationMode))
		mode_part = int(embedding % len(CapitalizationMode))
		return PoSCapitalizationMode(Pos(pos_part), CapitalizationMode(mode_part))


class StructureFeatureAnalyzer(object):
	NUM_FEATURES = len(Pos) * len(CapitalizationMode)

	@staticmethod
	def analyze(token: Token, mode: CapitalizationMode):
		pos = Pos.from_token(token)
		mode = PoSCapitalizationMode(pos, mode)
		return mode.to_embedding()




class StructureModel(object):
	SEQUENCE_LENGTH = MAX_SEQUENCE_LEN

 
	def __init__(self, use_gpu: bool = False):
		import tensorflow as tf
		from tensorflow.keras.models import Sequential
		from tensorflow.keras.layers import Dense, Embedding, LSTM
		from typing import List
		from tensorflow.keras.preprocessing.sequence import pad_sequences
		self.pad_sequences = pad_sequences
  
		latent_dim = StructureModel.SEQUENCE_LENGTH * 16

		with tf.device('/gpu:0' if use_gpu else '/cpu:0'):
			model = Sequential([
				Embedding(StructureFeatureAnalyzer.NUM_FEATURES, StructureFeatureAnalyzer.NUM_FEATURES),
				LSTM(latent_dim, dropout=0.2, return_sequences=False),
				Dense(StructureFeatureAnalyzer.NUM_FEATURES, activation='softmax')
			])

			model.build(input_shape=(None, StructureModel.SEQUENCE_LENGTH))
			model.summary()
			model.compile(loss='sparse_categorical_crossentropy', optimizer='adam')
			self.model = model

	def train(self, data, labels, epochs=1, validation_split=0.2):
		val_size = int(len(data) * validation_split)
		
		indices = np.arange(len(data))
		np.random.shuffle(indices)
		
		val_indices = indices[:val_size]
		train_indices = indices[val_size:]
		
		train_data, val_data = data[train_indices], data[val_indices]
		train_labels, val_labels = labels[train_indices], labels[val_indices]

		print(train_data[0], train_labels[0])
		
		self.model.fit(
			train_data,
			train_labels,
			epochs=epochs,
			batch_size=32,
			validation_data=(val_data, val_labels)
		)

	#TODOCAL: conditionally generation base on inputka
	def predict(self, num_sentences: int, struct_temp: float) -> List[PoSCapitalizationMode]:
		predictions = []
		sequence = [[0]]
		eos_count = 0

		idx = 0
		while eos_count < num_sentences:
			padded_sequence = self.pad_sequences(sequence, maxlen=StructureModel.SEQUENCE_LENGTH, padding='post')
			prediction = self.model.predict(padded_sequence, batch_size=1, verbose=0)[0]
			index = temp(prediction, struct_temp)
			if PoSCapitalizationMode.from_embedding(index).pos == Pos.EOS:
				eos_count += 1
			predictions.append(index)
			sequence[0].append(index)
			sequence[0] = sequence[0][-StructureModel.SEQUENCE_LENGTH:]
			idx = idx + 1
			#if idx > 25:
				#print("[structure] XDD TOO MANY TIRES")
				#break

		modes = [PoSCapitalizationMode.from_embedding(embedding) for embedding in predictions]
		#print(modes)
		return modes

	def load(self, path):
		self.model.load_weights(path)

	def save(self, path):
		self.model.save_weights(path)

class StructureModelWorker(MLModelWorker):
	def __init__(self, read_queue: Queue, write_queue: Queue, use_gpu: bool = False):
		MLModelWorker.__init__(self, name='SentenceStructureModelWorker', read_queue=read_queue,
							   write_queue=write_queue,
							   use_gpu=use_gpu)

	def run(self):
		self._model = StructureModel(use_gpu=self._use_gpu)
		MLModelWorker.run(self)

	def predict(self, *data, struct_temp) -> List[PoSCapitalizationMode]:
		return self._model.predict(num_sentences=data[0][0], struct_temp=struct_temp)

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

	def predict(self, num_sentences: int, struct_temp: float):
		return self._predict(num_sentences, struct_temp=struct_temp)

	def train(self, data, labels, epochs=1):
		return self._train(data, labels, epochs)

	def save(self, path):
		return self._save(path)

	def load(self, path):
		return self._load(path)
