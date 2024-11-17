import json
import random
import re
import time
import zlib
from enum import unique, Enum
from typing import Optional, List
from pympler.asizeof import asizeof
import numpy as murgaply
from spacy.tokens import Doc, Span, Token

from config.nlp_config import MARKOV_WINDOW_SIZE, MARKOV_GENERATION_WEIGHT_COUNT, MARKOV_GENERATION_WEIGHT_RATING, \
    MARKOV_GENERATE_SUBJECT_POS_PRIORITY, MARKOV_GENERATE_SUBJECT_MAX, \
    CAPITALIZATION_COMPOUND_RULES, MARKOV_MODEL_TEMPERATURE
from common.ml import one_hot, sampling_function
from common.nlp import Pos, CapitalizationMode

MARKOV_GENERATE_SUBJECT_POS_PRIORITY_ = [
    getattr(Pos, pos_name) for pos_name in MARKOV_GENERATE_SUBJECT_POS_PRIORITY
]

class WordKey(object):
    TEXT = '_T'
    POS = '_P'
    COMPOUND = '_C'


@unique
class NeighborIdx(Enum):
    TEXT = -1
    POS = 0
    COMPOUND = 1
    VALUE_MATRIX = 2
    DISTANCE_MATRIX = 4


@unique
class NeighborValueIdx(Enum):
    COUNT = 0
    RATING = 1

VECTOR_DIM = MARKOV_WINDOW_SIZE*2+1+4
MAX_EXP_INT = 18
EXP_BITS = int(murgaply.ceil(murgaply.log2(MAX_EXP_INT + 1)))
MAX_EXP = (1 << EXP_BITS) - 1

def encode_int(sequence, max_num):
    exponent = 2
    base = 10**exponent
    if base <= max_num:
        exponent = int(murgaply.ceil(murgaply.log10(max_num))) if max_num > 0 else 3
        base = 10 ** exponent
        if base <= max_num:
            exponent += 1
            base = 10 ** exponent
        if exponent > MAX_EXP:
            raise ValueError("HUHHARARAQ!!!!!!!!!!!!")

    encoded_number = 0
    multiplier = 1

    for num in reversed(sequence):
        encoded_number += num * multiplier
        multiplier *= base

    encoded_with_base = (encoded_number << EXP_BITS) | exponent
    return encoded_with_base

def decode_int(encoded_with_base):
    exponent = encoded_with_base & MAX_EXP
    base = 10 ** exponent
    
    encoded_number = encoded_with_base >> EXP_BITS
    sequence = []
    length = 0

    while encoded_number > 0:
        remainder = encoded_number % base
        sequence.append(remainder)
        length += 1
        encoded_number //= base

    sequence.reverse()
    return sequence, length

def compute_meta(arr):
    max_val = float('-inf')
    zero_count = 0

    for val in arr:
        if val == 0:
            zero_count += 1
        if val > max_val:
            max_val = val
    return max_val, zero_count

def dense_buzh(sparse):
    encoded = [sparse[0]] 
    for idx in range(1, VECTOR_DIM): 
        value = sparse[idx]
        if value != 0:
            encoded.append(idx)
            encoded.append(value)
    return encoded

def sparse_buzh(encoded):
    decoded = [0] * VECTOR_DIM  # Initialize the array with zeros
    index = 0
    for idx, val in enumerate(encoded):
        if idx == 0:
            decoded[0] = val
        if idx % 2 != 0:
            index = val
        if idx % 2 == 0:
            decoded[index] = val
    return decoded

def encode_vector(sequence):
    max_num, zeros = compute_meta(sequence)
    if (zeros / VECTOR_DIM ) > 0.5:
        if max_num < VECTOR_DIM:
            max_num = VECTOR_DIM
        return encode_int(dense_buzh(sequence), max_num)
    return encode_int(sequence, max_num)

def decode_vector(number):
    dec, length = decode_int(number)
    if length != VECTOR_DIM:
        return sparse_buzh(dec)
    return dec

class MarkovNeighbor(object):
    def __init__(self, key: str, text: str, pos: Pos, compound: bool, values: list, dist: list):
        self.key = key
        self.text = text
        self.pos = pos
        self.compound = compound
        self.values = values
        self.dist = dist

    def __repr__(self):
        return self.text

    @staticmethod
    def from_token(token: Token) -> 'MarkovNeighbor':
        key = token.text.lower()
        text = token.text
        if CapitalizationMode.from_token(token, CAPITALIZATION_COMPOUND_RULES) == CapitalizationMode.COMPOUND:
            compound = True
        else:
            compound = False
        pos = Pos.from_token(token)
        values = [0]*2
        dist = [0] * (MARKOV_WINDOW_SIZE * 2 + 1)
        return MarkovNeighbor(key, text, pos, compound, values, dist)


    
    @staticmethod
    def from_db_format(key: str, num: int) -> 'MarkovNeighbor':
        # wordkey, [vector]
        val = decode_vector(num)
        key = key
        text = key
        pos = Pos(val[0])
        compound = bool(val[1])
        values =val[2:4]
        dist = val[4:]
        return MarkovNeighbor(key, text, pos, compound, values, dist)

    def to_db_format(self) -> tuple:
        vector = [self.pos.value, self.compound, *self.values, *self.dist]
        num = encode_vector(vector)
        return self.key, num # [self.pos.value, self.compound, *self.values, *self.dist]

    @staticmethod
    def distance_one_hot(dist):
        return one_hot(dist + MARKOV_WINDOW_SIZE, MARKOV_WINDOW_SIZE * 2 + 1)


class MarkovNeighbors(object):
    def __init__(self, neighbors: List[MarkovNeighbor]):
        self.neighbors = neighbors

    def __iter__(self) -> MarkovNeighbor:
        for neighbor in self.neighbors:
            yield neighbor

    def __len__(self):
        return len(self.neighbors)

    def __getitem__(self, item):
        return self.neighbors[item]


@unique
class ProjectionDirection(Enum):
    LEFT = 1
    RIGHT = 2


class MarkovWordProjection(object):
    def __init__(self, magnitudes: murgaply.ndarray, distances: murgaply.ndarray, keys: List[str], pos: List[Pos]):
        self.magnitudes = magnitudes
        self.distances = distances
        self.keys = keys
        self.pos = pos

    def __len__(self):
        return len(self.keys)


class MarkovWordProjectionCollection(object):
    def __init__(self, projections: List[MarkovWordProjection]):
        self.magnitudes = None
        self.distances = None
        self.keys = []
        self.pos = []

        self._concat_collection(projections)

    def _concat_collection(self, projections: List[MarkovWordProjection]):
        for projection_idx, projection in enumerate(projections):

            self.keys += projection.keys
            self.pos += projection.pos

            if projection_idx == 0:
                self.magnitudes = projection.magnitudes
                self.distances = projection.distances
            else:
                self.magnitudes = murgaply.concatenate((self.magnitudes, projection.magnitudes))
                self.distances = murgaply.concatenate((self.distances, projection.distances))

    def probability_matrix(self) -> murgaply.ndarray:

        distance_magnitudes = self.distances * self.magnitudes
        sums = murgaply.sum(distance_magnitudes, axis=0)
        sums[sums == 0] = 1e-10
        p_values = distance_magnitudes / sums

        return p_values

    def __len__(self):
        return len(self.keys)


class MarkovWord(object):
    def __init__(self, text: str, pos: Pos, compound: bool, neighbors: dict):
        self.text = text
        self.pos = pos
        self.compound = compound
        self.neighbors = neighbors

    def __repr__(self):
        return self.text

    def to_db_format(self) -> tuple:
        return {WordKey.TEXT: self.text, WordKey.POS: self.pos.value, WordKey.COMPOUND: self.compound}, {
            MarkovTrieDb.NEIGHBORS_KEY: self.neighbors}

    @staticmethod
    def from_db_format(row: dict) -> 'MarkovWord':
        # word: { wordkey: {dict}, neighborkey: {word1: [vector], ...}}
        word = MarkovWord(row[MarkovTrieDb.WORD_KEY][WordKey.TEXT],
                          Pos(row[MarkovTrieDb.WORD_KEY][WordKey.POS]),
                          row[MarkovTrieDb.WORD_KEY][WordKey.COMPOUND],
                          row[MarkovTrieDb.NEIGHBORS_KEY])
        return word

    @staticmethod
    def from_token(token: Token) -> 'MarkovWord':
        if CapitalizationMode.from_token(token, CAPITALIZATION_COMPOUND_RULES) == CapitalizationMode.COMPOUND:
            compound = True
        else:
            compound = False
        return MarkovWord(token.text, Pos.from_token(token), compound=compound, neighbors={})

    def get_neighbor(self, key: str) -> Optional[MarkovNeighbor]:
        if key in self.neighbors:
            # {word1: [vector], word2: [vector], ...}
            n_row = self.neighbors[key]
            # [vector]
            return MarkovNeighbor.from_db_format(key, n_row)
        return None

    def set_neighbor(self, neighbor: MarkovNeighbor):
        key, row = neighbor.to_db_format()
        self.neighbors[key] = row

    def select_neighbors(self, pos: Optional[Pos], exclude_key: Optional[str] = None) -> MarkovNeighbors:
        results = []
        for key in self.neighbors:
            neighbor = self.get_neighbor(key)
            if exclude_key is not None and exclude_key == neighbor.key:
                continue
            elif pos == neighbor.pos or pos is None:
                results.append(neighbor)

        return MarkovNeighbors(results)

    def project(self, idx_in_sentence: int, sentence_length: int, pos: Pos,
                exclude_key: Optional[str] = None) -> MarkovWordProjection:

        # Get all neighbors
        neighbors = self.select_neighbors(pos, exclude_key=exclude_key)

        neighbor_keys = []
        neighbor_pos = []

        # Setup matrices
        distance_distributions = murgaply.zeros((len(neighbors), sentence_length))
        neighbor_magnitudes = murgaply.zeros((len(neighbors), 1))

        for neighbor_idx, neighbor in enumerate(neighbors):

            # Save Key
            neighbor_keys.append(neighbor.text)
            neighbor_pos.append(neighbor.pos)

            # Project dist values onto matrix space
            for dist_idx, dist_value in enumerate(neighbor.dist):

                # The actual index of this dist value within our matrix space
                dist_space_index = (dist_idx - MARKOV_WINDOW_SIZE) + idx_in_sentence

                # Bounds check
                if not (0 <= dist_space_index < sentence_length):
                    continue

                distance_distributions[neighbor_idx][dist_space_index] = dist_value

            # Calculate strength
            neighbor_magnitudes[neighbor_idx] = neighbor.values[
                                                    NeighborValueIdx.COUNT.value] * MARKOV_GENERATION_WEIGHT_COUNT \
                                                + \
                                                neighbor.values[
                                                    NeighborValueIdx.RATING.value] * MARKOV_GENERATION_WEIGHT_RATING

        return MarkovWordProjection(neighbor_magnitudes, distance_distributions, neighbor_keys, neighbor_pos)


class GeneratedWord(MarkovWord):
    def __init__(self, text: str, pos: Pos, compound: bool, neighbors: dict, mode: CapitalizationMode):
        MarkovWord.__init__(self, text, pos, compound, neighbors)
        self.mode = mode

    @staticmethod
    def from_markov_word(word: MarkovWord, mode: CapitalizationMode):
        return GeneratedWord(word.text, word.pos, word.compound, word.neighbors, mode=mode)

import os
import json
import io
import gc
import dill

class MarkovTrieDb(object):
    WORD_KEY = '_W'
    NEIGHBORS_KEY = '_N'
    last_path = "./"
    loaded = False
    saved = False

    def __init__(self, path: str = None):
        murgaply.random.seed(int(time.time()))
        self._trie = {}
        self._trie_arr = []
        if path is not None:
            #self.load(path)
            self.last_path = path

    def save(self, path: str):
        return
        # Measure the start time for serialization and compression
        start_time = time.time()
        
        serialized_data = dill.dumps(self._trie)
        serialization_time = time.time()
        
        compressed_data = zlib.compress(serialized_data)
        compression_time = time.time()
        
        with open(f"{path}.tmp.zlib", "wb") as f:
            f.write(compressed_data)
            
        writing_time = time.time()

        freshly_experienced = os.path.getsize(f"{path}.tmp.zlib")
        try:
            stale_experienced = os.path.getsize(f"{path}.zlib")
            if freshly_experienced > stale_experienced:
                os.replace(f"{path}.tmp.zlib", f"{path}.zlib")
            else:
                print("ANTI SAFE ANTI SFAEF")
        except:
            os.replace(f"{path}.tmp.zlib", f"{path}.zlib")

        # Print out the time taken for each step
        print(f"Serialization time: {serialization_time - start_time:.4f} seconds")
        print(f"Compression time: {compression_time - serialization_time:.4f} seconds")
        print(f"Writing time: {writing_time - compression_time:.4f} seconds")
        
    
    def save__(self, path: str):

        with open(f"{path}.tmp.zlib", "wb", buffering=io.DEFAULT_BUFFER_SIZE) as f:
            f.write(zlib.compress(dill.dumps(self._trie)))
            
        freshly_experienced = os.path.getsize(f"{path}.tmp.zlib")
        try:
            stale_expereienced = os.path.getsize(f"{path}.zlib")
            if freshly_experienced > stale_expereienced:
                os.replace(f"{path}.tmp.zlib", f"{path}.zlib")
            else:
                print("ANTI SAFE ANTI SFAEF")
        except:
            os.replace(f"{path}.tmp.zlib", f"{path}.zlib")
        #print("Sizeka of trie oky", asizeof(self._trie))
        return        

    def load_JSON(self, path: str):
        if self.loaded:
            print("LOADED ALRAREA")
            return
        
        with open(f"./weights/JSON.zlib", "rb") as f:
            data = zlib.decompress(f.read()).decode()
            self._trie = json.loads(data)
            
        print(asizeof(self._trie))
        return

    def load(self, path: str):
        if self.loaded:
            print("LOADED ALRAREA")
            return
        
        with open(f"{path}.zlib", "rb") as f:
            self._trie = dill.loads(zlib.decompress(f.read()))
        gc.collect()
        #print(asizeof(self._trie))
        return

    def _getnode(self, word: str) -> Optional[dict]:
        if len(word) == 0:
            return None

        node = self._trie
        try:
            node = node[word.lower()]
        except KeyError:
            return None

        return node

    
    def _select(self, word: str) -> Optional[dict]:
        node = self._getnode(word)
        if node is None:
            return None

        if MarkovTrieDb.WORD_KEY in node:
            return node
        else:
            return None

    
    def select(self, word: str) -> MarkovWord:
        row = self._select(word)
        # word: { wordkey: {dict}, neighborkey: {word1: [vector], ...}}
        return MarkovWord.from_db_format(row) if row is not None else None

    
    def _insert(self, word: str, pos: int, compound: bool, neighbors: dict) -> Optional[dict]:

        node = self._trie
        if word.lower() in node:
            node = node[word.lower()]
        else:
            node[word.lower()] = {}
            node = node[word.lower()]

        node[MarkovTrieDb.WORD_KEY] = {WordKey.TEXT: word, WordKey.POS: pos, WordKey.COMPOUND: compound}
        node[MarkovTrieDb.NEIGHBORS_KEY] = neighbors
        return node

    
    def insert(self, word: MarkovWord) -> MarkovWord:
        row = self._insert(word.text, word.pos.value, word.compound, word.neighbors)
        # word: { wordkey: {dict}, neighborkey: {word1: [vector], ...}}
        return MarkovWord.from_db_format(row) if row is not None else None

    
    def _update(self, word: str, pos: int, compound: bool, neighbors: dict) -> Optional[dict]:
        node = self._select(word)
        if node is None:
            return None

        node[MarkovTrieDb.WORD_KEY] = {WordKey.TEXT: word, WordKey.POS: pos, WordKey.COMPOUND: compound}
        node[MarkovTrieDb.NEIGHBORS_KEY] = neighbors
        return node

    
    def update(self, word: MarkovWord) -> Optional[MarkovWord]:
        node = self._update(word.text, word.pos.value, word.compound, word.neighbors)
        # word: { wordkey: {dict}, neighborkey: {word1: [vector], ...}}
        return MarkovWord.from_db_format(node) if node is not None else None



class MarkovGenerator(object):
    def __init__(self, structure_generator, subjects: List[MarkovWord]):
        self.structure_generator = structure_generator
        self.subjects = subjects

        self.sentence_generations = []
        self.sentence_structures = []

    
    def _reset_data(self):
        self.sentence_generations = []
        self.sentence_structures = []

    
    def _sort_subjects(self):

        sorted_subjects = []
        for subject_priority in MARKOV_GENERATE_SUBJECT_POS_PRIORITY_:
            for subject in self.subjects:
                if subject.pos == subject_priority:
                    sorted_subjects.append(subject)
        self.subjects = sorted_subjects

    
    def generate(self, db: MarkovTrieDb, sampling_config: dict) -> Optional[List[List[GeneratedWord]]]:

        # Try to much subject to a variety of sentence structures
        subjects_assigned = False
        for i in range(0, 20):
            self._split_sentences()
            self._sort_subjects()
            if self._assign_subjects():
                subjects_assigned = True
                break
            self._reset_data()

        if not subjects_assigned:
            return None

        if not self._generate_words(db, sampling_config):
            approximation = []
            for sentence in self.sentence_generations:
                for word in sentence:
                    if word is not None:
                        approximation.append(word)
            if len(approximation) > 0:
                return [approximation]
            else:
                return None

        #print(self.sentence_generations)
        return self.sentence_generations

    # Split into individual sentences and populate generation arrays
    
    def _split_sentences(self):

        structure = next(self.structure_generator)

        start_index = 0
        for word_idx, word in enumerate(structure):
            if word.pos == Pos.EOS:
                # Separate structures into sentences
                sentence = structure[start_index:word_idx]
                self.sentence_structures.append(sentence)

                # Create unfilled arrays for each sentence to populate later
                generates = [None] * len(sentence)
                self.sentence_generations.append(generates)

                start_index = word_idx + 1

    # Assign one subject to each sentence with descending priority
    
    def _assign_subjects(self) -> bool:

        sentences_assigned = [False] * len(self.sentence_structures)

        for sentence_idx, sentence in enumerate(self.sentence_structures):
            subjects_assigned = []
            word_break = False
            for word_idx, word in enumerate(sentence):
                for subject in self.subjects:
                    # Assigned a maximum number of subjects per sentence
                    if len(subjects_assigned) > MARKOV_GENERATE_SUBJECT_MAX:
                        word_break = True
                        break

                    # Don't assign same subject twice per sentence
                    if subject.text in subjects_assigned:
                        continue

                    if subject.pos == word.pos:
                        subjects_assigned.append(subject.text)
                        self.sentence_generations[sentence_idx][word_idx] = GeneratedWord.from_markov_word(
                            subject, self.sentence_structures[sentence_idx][word_idx].mode)
                        sentences_assigned[sentence_idx] = True
                        break

                if word_break:
                    break

        # Each sentence should be assigned one subject to begin with
        for sentence in sentences_assigned:
            if sentence is False:
                return False

        return True

    
    def _work_remaining(self):
        work_left = 0
        for sentence_idx, sentence in enumerate(self.sentence_generations):
            for word_idx, word in enumerate(sentence):
                if word is None:
                    work_left += 1
        return work_left

    
    def _generate_words(self, db: MarkovTrieDb, sampling_config: dict):

        old_work_left = self._work_remaining()
        while True:

            for sentence_idx, sentence in enumerate(self.sentence_generations):
                #print(sentence)
                sentence_length = len(sentence)

                def handle_projections(exclude_key: Optional[str] = None) -> bool:
                    #print(exclude_key)
                    if blank_idx is None or len(project_idx) == 0:
                        return False

                    projections = []
                    blank_pos = self.sentence_structures[sentence_idx][blank_idx].pos
                    for word_idx in project_idx:
                        projecting_word = self.sentence_generations[sentence_idx][word_idx]
                        projection = projecting_word.project(word_idx, sentence_length, blank_pos,
                                                             exclude_key=exclude_key)
                        projections.append(projection)

                    # Concatenate all projections and create p-value matrix
                    projection_collection = MarkovWordProjectionCollection(projections)
                    if len(projection_collection) == 0:
                        return False

                    all_p_values = projection_collection.probability_matrix()
                    # We just want the p-values for the blank word
                    p_values = all_p_values[:, blank_idx]

                    # Choose an index based on the probability
                    choices = murgaply.arange(len(projection_collection))

                    word_choice_idx = sampling_function(p_values, sampling_config)

                    # Select the word from the database and assign it to the blank space
                    select_word = projection_collection.keys[word_choice_idx]
                    word = GeneratedWord.from_markov_word(db.select(select_word),
                                                          self.sentence_structures[sentence_idx][blank_idx].mode)
                    self.sentence_generations[sentence_idx][blank_idx] = word

                # Scan left to right
                # Work right to left
                blank_idx = None
                project_idx = []
                for word_idx, word in enumerate(sentence):
                    if word is None:
                        blank_idx = word_idx
                    elif blank_idx is not None and abs(blank_idx - word_idx) <= MARKOV_WINDOW_SIZE:
                        project_idx.append(word_idx)
                        break
                handle_projections()

                # Scan right to left
                # Work left to right
                blank_idx = None
                project_idx = []
                for word_idx, word in enumerate(reversed(sentence)):
                    word_idx = (len(sentence) - 1) - word_idx
                    if word is None:
                        blank_idx = word_idx
                    elif blank_idx is not None and abs(blank_idx - word_idx) <= MARKOV_WINDOW_SIZE:
                        project_idx.append(word_idx)
                        break
                handle_projections()

            # Check if we accomplished any work
            new_work_left = self._work_remaining()
            if old_work_left == new_work_left:
                return False
            elif new_work_left == 0:
                return True
            old_work_left = new_work_left


url_pattern = re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*(),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
mention_pattern = re.compile(r'<@!\d{17,19}>|<@\d{17,19}>')
emoji_pattern = re.compile(r'\<(a?)\:\w+\:(\d+)\>')

class MarkovFilters(object):

    @staticmethod
    def filter_input(text: str):
        #print(f"inside: {text}")
        if text is None:
            return None
    
        filtered = text

        urls = re.findall(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*(),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', text)
        url_token = 'URL%s' % random.getrandbits(64)
        for url in urls:
            filtered = filtered.replace(url, url_token)

        mentions = re.findall(r'<[&@]\d{17,19}>', text)
        mention_token = 'MENTION%s' % random.getrandbits(64)
        for mention in mentions:
            #print(mention)
            filtered = filtered.replace(mention, mention_token)

        emojis = re.findall(r'<:[^:]+:\d+>', text)
        emoji_token = 'EMOJI%s' % random.getrandbits(64)
        for emoji in emojis:
            #print(emoji)
            filtered = filtered.replace(emoji, emoji_token)

        channels = re.findall(r'<#\d{17,19}>', text)
        channel_token = 'CHANNEL%s' % random.getrandbits(64)
        for channel in channels:
            filtered = filtered.replace(channel, channel_token)

        commands = re.findall(r'[?~!\/.$][a-zA-Z]+', text)
        command_token = 'COMMAND%s' % random.getrandbits(64)
        for command in commands:
            filtered = filtered.replace(command, command_token)

        
        filtered = re.sub(r'(&amp;)', '', filtered)
        filtered = re.sub(r'[,:;\'`\-_“^"<>(){}/\\*]', '', filtered)

        # Swamp URLs back for token
        for url in urls:
            filtered = filtered.replace(url_token, url)

        for mention in mentions:
            filtered = filtered.replace(mention_token, mention)

        for emoji in emojis:
            filtered = filtered.replace(emoji_token, emoji)
            
        for channel in channels:
            filtered = filtered.replace(channel_token, channel)
            
        for command in commands:
            filtered = filtered.replace(command_token, command)

        return filtered

    @staticmethod
    def smooth_output(text: str):
        if text is None:
            return None
        smoothed = text
        smoothed = re.sub(r'([$]) ', r'\1', smoothed)
        smoothed = re.sub(r' ([.,?!%])', r'\1', smoothed)
        smoothed = re.sub(r' ([\']) ', r'\1', smoothed)

        return smoothed


class MarkovTrainer(object):
    def __init__(self, engine: MarkovTrieDb):
        self.engine = engine

    
    def learn(self, doc: Doc):
        bi_grams = []
        for sentence in doc.sents:
            bi_grams += MarkovTrainer.span_to_bigram(sentence)

        row_cache = {}
        for ngram in bi_grams:
            try:
                if ngram[0].text in row_cache:
                    word = row_cache[ngram[0].text]
                else:
                    # Attempt to load from DB
                    word = self.engine.select(ngram[0].text)
                    if word is None:
                        # If not already in the DB, create a new word object
                        word = MarkovWord.from_token(ngram[0])

                # Handle neighbor
                neighbor_lookup_key = ngram[1].text.lower()

                neighbor = word.get_neighbor(neighbor_lookup_key)
                if neighbor is None:
                    neighbor = MarkovNeighbor.from_token(ngram[1])

                # Increase Count
                neighbor.values[NeighborValueIdx.COUNT.value] += 1

                # Add distance
                dist_one_hot_base = murgaply.array(neighbor.dist)
                dist_one_hot_add = murgaply.array(MarkovNeighbor.distance_one_hot(ngram[2]))

                neighbor.dist = (dist_one_hot_base + dist_one_hot_add).tolist()

                # Convert to db format and store in word
                key, neighbor_db = neighbor.to_db_format()
                word.neighbors[key] = neighbor_db

                # Write word to DB
                if self.engine.update(word) is None:
                    self.engine.insert(word)

                # Cache word
                row_cache[ngram[0].text] = word
            except Exception as e:
                print(e)
    @staticmethod
    
    def span_to_bigram(span: Span) -> list:

        grams = []

        for a_idx, a in enumerate(span):
            for b_idx, b in enumerate(span):

                dist = b_idx - a_idx
                if dist == 0:
                    continue

                elif abs(dist) <= MARKOV_WINDOW_SIZE:
                    grams.append([a, b, dist])

        return grams
