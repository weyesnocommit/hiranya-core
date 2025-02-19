from markov_engine import MarkovTrieDb, MarkovFilters, MarkovGenerator, MarkovGeneratorBERT
from models.structure import StructureModelScheduler
from common.nlp import CapitalizationMode
from typing import Optional, List
from multiprocessing import Process, Queue, Event
from threading import Thread
from queue import Empty
from spacy.tokens import Doc
from storage.armchair_expert import InputTextStatManager
import numpy as np
import random
from config.nlp_config import MARKOV_MODEL_TEMPERATURE, STRUCTURE_MODEL_TEMPERATURE
from config.bot_config import MISUNDERSTOOD_LIST, UNHEARD_LIST

class ConnectorRecvMessage(object):
    def __init__(self, text: str, learn: bool=False, reply=True, sampling_config=None, ignore_topics = None):
        self.text = text
        self.learn = learn
        self.reply = reply
        if not sampling_config:
            self.sampling_config = {
                'markov': {
                    'strategy': 'softmax',
                    'temperature': MARKOV_MODEL_TEMPERATURE
                },
                'struct': {
                    'temperature': STRUCTURE_MODEL_TEMPERATURE
                }
            }
        else:
            self.sampling_config = sampling_config
        if ignore_topics:
            self.ignore_topics = ignore_topics
        else:
            self.ignore_topics = []

class ConnectorReplyGenerator(object):
    def __init__(self, markov_model: MarkovTrieDb,
                 structure_scheduler: StructureModelScheduler = None,
                 use_bert: bool = False):  # Toggle for BERT
        self._markov_model = markov_model
        self._structure_scheduler = structure_scheduler
        self._nlp = None
        self._use_bert = use_bert  # Toggle between LSTM and BERT

    def give_nlp(self, nlp):
        self._nlp = nlp

    def generate(self, message: str, sampling_config: dict, doc: Doc = None, ignore_topics=None) -> Optional[str]:
        if ignore_topics is None:
            ignore_topics = []
        print(sampling_config)
        if self._use_bert:
            # BERT-based generation (no structure generator or NLP preprocessing needed)
            return self._generate_with_bert(message, sampling_config)
        else:
            # LSTM-based generation (uses structure generator and NLP preprocessing)
            return self._generate_with_lstm(message, sampling_config, doc, ignore_topics)

    def _generate_with_lstm(self, message: str, sampling_config: dict, doc: Doc = None, ignore_topics= None) -> Optional[str]:
        """Generate a reply using the LSTM-based approach."""
        filtered_message = "Huhharabin"
        if doc is None:
            filtered_message = MarkovFilters.filter_input(message)
            doc = self._nlp(filtered_message)

        subjects = []
        for token in doc:
            if token.text in ignore_topics:
                continue
            markov_word = self._markov_model.select(token.text)
            if markov_word is not None:
                subjects.append(markov_word)
        if len(subjects) == 0:
            return random.choice(UNHEARD_LIST)

        def structure_generator():
            sentence_stats_manager = InputTextStatManager()
            while True:
                choices, p_values = sentence_stats_manager.probabilities()
                if len(choices) > 0:
                    num_sentences = np.random.choice(choices, p=p_values)
                else:
                    num_sentences = np.random.randint(1, 10)
                yield self._structure_scheduler.predict(num_sentences=num_sentences, sampling_config=sampling_config['struct'])

        generator = MarkovGenerator(structure_generator=structure_generator(), subjects=subjects)

        reply_words = []
        sentences = generator.generate(db=self._markov_model, sampling_config=sampling_config['markov'])
        if sentences is None:
            return random.choice(MISUNDERSTOOD_LIST)

        for sentence in sentences:
            for word_idx, word in enumerate(sentence):
                if not word.compound:
                    text = CapitalizationMode.transform(word.mode, word.text)
                else:
                    text = word.text
                reply_words.append(text)

        reply = " ".join(reply_words)
        return MarkovFilters.smooth_output(reply)

    def _generate_with_bert(self, message: str, sampling_config: dict) -> Optional[str]:
        """Generate a reply using BERT embeddings."""
        sampling_config = sampling_config['markov']
        # Use BERT to guide the generation process
        generator = MarkovGeneratorBERT()  # No need for subjects in BERT mode
        reply_words = generator.generate_with_bert(db=self._markov_model, sampling_config=sampling_config, context=message)

        if not reply_words:
            return random.choice(MISUNDERSTOOD_LIST)

        reply = " ".join(reply_words)
        return MarkovFilters.smooth_output(reply)


class ConnectorWorker(Process):
    def __init__(self, name, read_queue: Queue, write_queue: Queue, shutdown_event: Event):
        Process.__init__(self, name=name)
        self._read_queue = read_queue
        self._write_queue = write_queue
        self._shutdown_event = shutdown_event
        self._frontend = None

    def send(self, message: ConnectorRecvMessage):
        return self._write_queue.put(message)

    def recv(self) -> Optional[str]:
        return self._read_queue.get()

    def run(self):
        pass


class ConnectorScheduler(object):
    def __init__(self, shutdown_event: Event):
        self._read_queue = Queue()
        self._write_queue = Queue()
        self._shutdown_event = shutdown_event
        self._worker = None

    def recv(self, timeout: Optional[float]) -> Optional[ConnectorRecvMessage]:
        try:
            return self._read_queue.get(timeout=timeout)
        except Empty:
            return None

    def send(self, message: str):
        self._write_queue.put(message)

    def start(self):
        self._worker.start()

    def shutdown(self):
        self._worker.join()


class Connector(object):
    def __init__(self, reply_generator: ConnectorReplyGenerator, connectors_event: Event):
        self._reply_generator = reply_generator
        self._scheduler = None
        self._thread = Thread(target=self.run)
        self._write_queue = Queue()
        self._read_queue = Queue()
        self._frontends_event = connectors_event
        self._shutdown_event = Event()
        self._muted = True

    def give_nlp(self, nlp):
        self._reply_generator.give_nlp(nlp)

    def start(self):
        self._scheduler.start()
        self._thread.start()

    def run(self):
        while not self._shutdown_event.is_set():
            message = self._scheduler.recv(timeout=0.2)
            if self._muted:
                self._scheduler.send(None)
            elif message is not None:
                # Receive the message and put it in a queue
                self._read_queue.put(message)
                # Notify main program to wakeup and check for messages
                self._frontends_event.set()
                # Send the reply
                reply = self._write_queue.get()
                self._scheduler.send(reply)

    def send(self, message: str):
        self._write_queue.put(message)

    def recv(self) -> Optional[ConnectorRecvMessage]:
        if not self._read_queue.empty():
            return self._read_queue.get()
        return None

    def shutdown(self):
        # Shutdown event signals both our thread and process to shutdown
        self._shutdown_event.set()
        self._scheduler.shutdown()
        self._thread.join()

    def generate(self, message: str, sampling_config: dict, doc: Doc=None, ignore_topics = None) -> str:
        return self._reply_generator.generate(message, sampling_config, doc, ignore_topics)

    def mute(self):
        self._muted = True

    def unmute(self):
        self._muted = False

    def empty(self):
        return self._read_queue.empty()
