import zlib
import dill
import argparse
import logging
import signal
import sys
import traceback
import os
from enum import Enum, unique
from multiprocessing import Event
#import numpy as np
from common.nlp import create_nlp_instance, SpacyPreprocessor
from config.bot_config import ARMCHAIR_EXPERT_LOGLEVEL
from config.nlp_config import USE_GPU, STRUCTURE_MODEL_PATH, MARKOV_DB_PATH, STRUCTURE_MODEL_TRAINING_MAX_SIZE, MARKOV_MODEL_MAX_PREPROCESSING_SIZE
from markov_engine import MarkovTrieDb, MarkovTrainer, MarkovFilters
from models.structure import StructureModelScheduler, StructurePreprocessor
from storage.armchair_expert import InputTextStatManager
from storage.imported import ImportTrainingDataManager
from spacy.tokens import DocBin


@unique
class AEStatus(Enum):
    STARTING_UP = 1
    RUNNING = 2
    SHUTTING_DOWN = 3
    SHUTDOWN = 4


class HiranyaCore(object):
    def __init__(self):
        # Placeholders
        self._markov_model = None
        self._nlp = None
        self._status = None
        self._structure_scheduler = None
        self._connectors = []
        self._connectors_event = Event()
        self._zmq_connector = None
        self._logger = logging.getLogger(self.__class__.__name__)
        
    def _set_status(self, status: AEStatus):
        self._status = status
        self._logger.info("Status: %s" % str(self._status).split(".")[1])

    def start(self, retrain_structure: bool = False, retrain_markov: bool = False):

        self._set_status(AEStatus.STARTING_UP)
        # Markobaka
        self._markov_model = MarkovTrieDb(MARKOV_DB_PATH)
        if not retrain_markov:
            try:
                self._markov_model.load(MARKOV_DB_PATH)
                print("Info: M load markov db ok)")
            except FileNotFoundError as e:
                print(e)
                retrain_markov = True
                
        # struct
        self._structure_scheduler = StructureModelScheduler(USE_GPU)
        self._structure_scheduler.start()
        structure_model_trained = None
        if not retrain_structure is None:
            try:
                open(STRUCTURE_MODEL_PATH, 'rb')
                self._structure_scheduler.load(STRUCTURE_MODEL_PATH)
                structure_model_trained = True
            except FileNotFoundError:
                structure_model_trained = False

        try:
            from connectors.server import ZmqFrontend, DiscordReplyGenerator
            zmq_reply_generator = DiscordReplyGenerator(markov_model=self._markov_model,
                                                           structure_scheduler=self._structure_scheduler)
            self._zmq_connector = ZmqFrontend(reply_generator=zmq_reply_generator,
                                                      connectors_event=self._connectors_event)
            self._connectors.append(self._zmq_connector)
            self._logger.info("Loaded Discord Connector.")
            
            
        except Exception as e:
            print(e)
            pass
            

        # Non forking initializations
        self._logger.info("Loading spaCy model")
        self._nlp = create_nlp_instance()

        # Catch up on training now that everything is initialized but not yet started
        if retrain_structure or not structure_model_trained:
            self.train(retrain_structure=True, retrain_markov=retrain_markov)
        else:
            self.train(retrain_structure=False, retrain_markov=retrain_markov)

        # Give the connectors the NLP object and start them
        for connector in self._connectors:
            connector.give_nlp(self._nlp)
            connector.start()
            connector.unmute()

        # Handle events
        self._main()

    
    def _preprocess_structure_data(self):
        structure_preprocessor = StructurePreprocessor()
        db_path = './db/docbin.spacy.zlib'
        docbin = DocBin()
        processed_docs = {}
        #to_add = []

        #if os.path.exists(db_path):
        #    with open(db_path, 'rb') as f:
        #        docbin = DocBin().from_bytes(f.read())
        #    for doc in docbin.get_docs(self._nlp.vocab):
        #        processed_docs[doc.text] = doc

        self._logger.info("PREPROCESSKA STRUCTURE (Import)")
        imported_messages = ImportTrainingDataManager().all_training_data(
            limit=STRUCTURE_MODEL_TRAINING_MAX_SIZE,
            order_by='id', order='desc'
        )
        print(len(imported_messages))
        for message_idx, message in enumerate(imported_messages):
            if message_idx % 100 == 0:
                self._logger.info(f"Preprocesska (Import): {(message_idx / min(STRUCTURE_MODEL_TRAINING_MAX_SIZE, len(imported_messages)) * 100):.2f}%")

            message_text = MarkovFilters.filter_input(message[1].decode('utf-8'))
            try:
                if message_text in processed_docs:
                    doc = processed_docs[message_text]
                    if not structure_preprocessor.preprocess(doc):
                        return structure_preprocessor
                else:
                    doc = self._nlp(message_text)
                    #processed_docs[doc.text] = doc
                    #to_add.append(doc)
                    if not structure_preprocessor.preprocess(doc):
                        return structure_preprocessor
            except Exception as e:
                print(e)
                pass

        discord_messages = None
        if self._zmq_connector is not None:
            self._logger.info("PREPROCESSKA STRUCTURE (Discord)")
            from storage.message_storage import DiscordTrainingDataManager

            discord_messages = DiscordTrainingDataManager().all_training_data(
                limit=STRUCTURE_MODEL_TRAINING_MAX_SIZE,
                order_by='id', order='desc'
            )

            for message_idx, message in enumerate(discord_messages):
                if message_idx % 100 == 0:
                    self._logger.info(f"preprocesska (Discord):{(message_idx / min(STRUCTURE_MODEL_TRAINING_MAX_SIZE, len(discord_messages)) * 100):.2f}%")

                message_text = MarkovFilters.filter_input(message[1].decode('utf-8'))
                try:
                    if message_text in processed_docs:
                        doc = processed_docs[message_text]
                        if not structure_preprocessor.preprocess(doc):
                            return structure_preprocessor
                    else:
                        doc = self._nlp(message_text)
                        #processed_docs[doc.text] = doc
                        #to_add.append(doc)
                        if not structure_preprocessor.preprocess(doc):
                            return structure_preprocessor
                except Exception as e:
                    print(e)
                    pass


        #for doc_idx, doc in enumerate(to_add):
            #if doc_idx % 100 == 0:
            #    self._logger.info(f"Docbiinka baaritanka: {(doc_idx / len(to_add) * 100):.2f}%")
            #docbin.add(doc)

        #with open(db_path, 'wb') as f:
        #    f.write(docbin.to_bytes())


        return structure_preprocessor
    

    def _save_gzipped_dill(self, obj, file_path):
        return 
        with open(file_path, "wb") as f:
            f.write(zlib.compress(obj))

    def _load_gzipped_dill(self, file_path):
        return DocBin()
        with open(file_path, "rb") as f:
            return dill.loads(zlib.decompress(f.read()))

    def _preprocess_markov_data(self, all_training_data: bool = False):
        spacy_preprocessor = SpacyPreprocessor()
        itd = ImportTrainingDataManager()
        db_path = './db/docbin.cache.gz'
        docbin = DocBin()
        processed_docs = {}

        #if os.path.exists(db_path):
        #    docbin = self._load_gzipped_dill(db_path)
        #    for doc in docbin.get_docs(self._nlp.vocab):
        #        processed_docs[doc.text] = doc

        if all_training_data:
            itd.mark_untrained()
        iids = []
        dids = []
        self._logger.info("PREPROCESSSKA MARKOV (Import)")
        imported_messages = itd.all_training_data(
            limit=MARKOV_MODEL_MAX_PREPROCESSING_SIZE,
            order_by='id', order='desc'
        )
        self._logger.info(f"Fetched {len(imported_messages)} messages (Import)")

        for message_idx, message in enumerate(imported_messages):
            iids.append(message[0])
            message_text = MarkovFilters.filter_input(message[1].decode('utf-8'))

            if message_idx % 100 == 0:
                self._logger.info(f"preprocesska (Import) {(message_idx / len(imported_messages) * 100):.2f}%")

            try:
                if message_text in processed_docs:
                    doc = processed_docs[message_text]
                    spacy_preprocessor.preprocess(doc)
                else:
                    doc = self._nlp(message_text)
                    spacy_preprocessor.preprocess(doc)
                    #docbin.add(doc)
            except Exception as e:
                self._logger.error(f"Error processing message {message_idx}: {e}")

        if self._zmq_connector is not None:
            self._logger.info("PREPROCESSSKA MARKOV (Discord)")
            from storage.message_storage import DiscordTrainingDataManager
            dtd = DiscordTrainingDataManager()
            if all_training_data:
                dtd.mark_untrained()
            discord_messages = dtd.all_training_data(
                limit=MARKOV_MODEL_MAX_PREPROCESSING_SIZE,
                order_by='id', order='desc'
            )
            self._logger.info(f"Fetched {len(discord_messages)} messages (Discord)")

            for message_idx, message in enumerate(discord_messages):
                dids.append(message[0])
                message_text = MarkovFilters.filter_input(message[1].decode('utf-8'))

                if message_idx % 100 == 0:
                    self._logger.info(f"preprocesska (Discord) {(message_idx / len(discord_messages) * 100):.2f}%")

                try:
                    if message_text in processed_docs:
                        doc = processed_docs[message_text]
                        spacy_preprocessor.preprocess(doc)
                    else:
                        doc = self._nlp(message_text)
                        spacy_preprocessor.preprocess(doc)
                        #docbin.add(doc)
                except Exception as e:
                    self._logger.error(f"Error processing Discord message {message_idx}: {e}")

        # Save the processed DocBin as a gzipped dilled object
        self._save_gzipped_dill(docbin, db_path)

        return spacy_preprocessor, iids, dids
    
    def _train_markov(self, retrain: bool = False):

        spacy_preprocessor, iids, dids = self._preprocess_markov_data(all_training_data=retrain)

        self._logger.info("Training(Markov)")
        input_text_stats_manager = InputTextStatManager()
        if retrain:
            # Reset stats if we are retraining
            input_text_stats_manager.reset()

        markov_trainer = MarkovTrainer(self._markov_model)
        docs, _ = spacy_preprocessor.get_preprocessed_data()
        for doc_idx, doc in enumerate(docs):
            # Print Progress
            if doc_idx % 100 == 0:
                self._logger.info("Training(Markov): %f%%" % (doc_idx / len(docs) * 100))

            markov_trainer.learn(doc)

            sents = 0
            for sent in doc.sents:
                sents += 1
            input_text_stats_manager.log_length(length=sents)

        if len(docs) > 0:
            self._markov_model.save(MARKOV_DB_PATH)
            input_text_stats_manager.commit()
        
        return iids, dids
            
    
    def _train_structure(self, retrain: bool = False):

        if not retrain:
            return

        structure_preprocessor = self._preprocess_structure_data()

        self._logger.info("Training(Structure)")
        structure_data, structure_labels = structure_preprocessor.get_preprocessed_data()
        if len(structure_data) > 0:
            print(structure_data[0], structure_labels[0])
            # I don't know anymore
            epochs = 3

            self._structure_scheduler.train(structure_data, structure_labels, epochs=epochs)
            self._structure_scheduler.save(STRUCTURE_MODEL_PATH)

    
    def train(self, retrain_structure: bool = False, retrain_markov: bool = False):

        self._logger.info("Training begin")
        iids = []
        dids = []
        if not retrain_structure:
            iids, dids = self._train_markov(retrain_markov)
        
        self._train_structure(retrain_structure)

        # Mark data as trained
        if self._zmq_connector is not None:
            from storage.message_storage import DiscordTrainingDataManager
            DiscordTrainingDataManager().mark_trained(dids)
        ImportTrainingDataManager().mark_trained(iids)
        self._logger.info("Training end")

    def _main(self):
        self._set_status(AEStatus.RUNNING)

        while True:
            if self._connectors_event.wait(timeout=1):
                self._connectors_event.clear()

            for connector in self._connectors:
                while not connector.empty():
                    message = connector.recv()
                    if message is not None and message.text:
                        try:
                            doc = self._nlp(MarkovFilters.filter_input(message.text))
                            if message.learn:
                                MarkovTrainer(self._markov_model).learn(doc)
                                connector.send(None)
                            if message.reply:
                                reply = connector.generate(
                                    message.text, 
                                    message.sampling_config,
                                    doc=doc, 
                                    ignore_topics=message.ignore_topics
                                )
                                connector.send(reply)
                        except Exception as e:
                            print(e)
                            print(traceback.format_exc())
                            connector.send(None)
                    else:
                        connector.send(None)

            if self._status == AEStatus.SHUTTING_DOWN:
                self.shutdown()
                self._set_status(AEStatus.SHUTDOWN)
                sys.exit(0)

    def shutdown(self):

        # Shutdown connectors
        for connector in self._connectors:
            connector.shutdown()

        # Shutdown models
        self._structure_scheduler.shutdown()

    def handle_shutdown(self):
        # Shutdown main()
        self._set_status(AEStatus.SHUTTING_DOWN)


def signal_handler(sig, frame):
    if sig == signal.SIGINT:
        cpe.handle_shutdown()


if __name__ == '__main__':
    sys.setrecursionlimit(2147483647)
    signal.signal(signal.SIGINT, signal_handler)
    logging.basicConfig(
            level=ARMCHAIR_EXPERT_LOGLEVEL,
            format='[%(asctime)s][%(levelname)s][%(name)s]: %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
    )

    parser = argparse.ArgumentParser()
    parser.add_argument('--retrain-markov', help='Retrain the markov word engine with all training data',
                        action='store_true')
    parser.add_argument('--retrain-structure', help='Retrain the structure RNN with all available training data',
                        action='store_true')
    args = parser.parse_args()

    cpe = HiranyaCore()
    cpe.start(retrain_structure=args.retrain_structure, retrain_markov=args.retrain_markov)
