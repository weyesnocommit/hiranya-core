import asyncio
import re
import logging
from connectors.connector_common import *
from config.bot_config import *
from spacy.tokens import Doc
import zmq
import asyncio
from queue import Queue
from threading import Event
import msgpack
import json

class DiscordReplyGenerator(ConnectorReplyGenerator):

    BAD_WORDS = {
        re.compile(r'nigger', re.IGNORECASE): 'RASCAL',
        re.compile(r'nigga', re.IGNORECASE): 'RASCAL',
        re.compile(r'niger', re.IGNORECASE): 'rascal',
        re.compile(r'niga', re.IGNORECASE): 'rascal',
        re.compile(r'ngr', re.IGNORECASE): 'rascal'
    }

    USER_MENTION_PATTERN = re.compile(r'@(\d{18,19})', re.IGNORECASE)
    EMOJI_PATTERN = re.compile(r'([A-Za-z0-9_]+):(\d{18,19})', re.IGNORECASE)
    BOT_MENTION_PATTERN = re.compile(r'@!(\d{18,19})', re.IGNORECASE)
    SPECIFIC_USER_PATTERN = re.compile(r'794890213977358337', re.IGNORECASE)
    CHANNEL_PATTERN = re.compile(r'795310503135412264', re.IGNORECASE)
    
    def generate(self, message: str, doc: Doc = None) -> Optional[str]:
        reply = super().generate(message, doc, ignore_topics=[""])
        
        if reply is None:
            return None

        # fix user and channel mentions
        reply = self.USER_MENTION_PATTERN.sub(r'<@\1>', reply)
        reply = self.EMOJI_PATTERN.sub(r'<:\1:\2>', reply)
        reply = self.BOT_MENTION_PATTERN.sub(r'<@!\1>', reply)
        reply = self.SPECIFIC_USER_PATTERN.sub('', reply)
        reply = self.CHANNEL_PATTERN.sub(r'<#795310503135412264>', reply)

        # Bad word filtering
        for bad_word_pattern, replacement in self.BAD_WORDS.items():
            reply = bad_word_pattern.sub(replacement, reply)

        # Optionally remove URLs
        if False:
            reply = re.sub(
                r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*(),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', reply)
            reply = re.sub(r' +', ' ', reply).strip()

        return reply if len(reply) > 0 else None


class ZmqWorker(ConnectorWorker):
    def __init__(self, read_queue: Queue, write_queue: Queue, shutdown_event: Event):
        ConnectorWorker.__init__(self, name='ZmqWorker', read_queue=read_queue, write_queue=write_queue,
                                 shutdown_event=shutdown_event)
        self._logger = None
        

    async def handle_message(self, message):
        try:
              # raw=False to get strings as UTF-8
            if message['type'] == 'ping':
                return ({'type': 'pong', 'from': 'markov'})
            self._logger.info(message)
            notknow = ConnectorRecvMessage(message['text'], message['learn'], message['reply'])
            self.send(notknow)
            reply = self.recv() 
            if message['store'] and message['dmessage']:
                self._db.store_dict(message['dmessage'])  # Store the deserialized object
            return reply
        except Exception as e:
            self._logger.error(e)
            return "HUUNGNANHJANFHAISFJ ASNFUASH WASGW FAM WHA WKD"

    async def zmq_server(self):
        while not self._shutdown_event.is_set():
            packed_message = None
            try:
                packed_message = self._socket.recv()
                message = msgpack.unpackb(packed_message)
                response = await self.handle_message(message)
                packed_response = msgpack.packb(response)
                self._socket.send(packed_response)
            except Exception as e:
                if "current state" in str(e):
                    self._socket.close()
                    self._socket = self._context.socket(zmq.REP)
                    self._socket.bind(f"tcp://127.0.0.1:{MARKOV_PORT}")
                self._logger.error(f"Error processing message: {str(e)}")
                self._logger.error(packed_message)
                
                
    def run(self):
        self._context = zmq.Context()
        self._socket = self._context.socket(zmq.REP)
        from storage.message_storage import DiscordTrainingDataManager
        logging.basicConfig(
            level=MARKOV_SERVER_LOG_LEVEL,
            format='[%(asctime)s][%(levelname)s][%(name)s]: %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        self._logger = logging.getLogger(self.__class__.__name__)
        self._db = DiscordTrainingDataManager()

        self._socket.bind(f"tcp://127.0.0.1:{MARKOV_PORT}")
        loop = asyncio.get_event_loop()
        try:
            loop.run_until_complete(self.zmq_server())
        finally:
            self._socket.close()
            self._context.term()

class ZmqScheduler(ConnectorScheduler):
    def __init__(self, shutdown_event: Event):
        ConnectorScheduler.__init__(self, shutdown_event)
        self._worker = ZmqWorker(read_queue=self._write_queue, write_queue=self._read_queue, shutdown_event=shutdown_event)

class ZmqFrontend(Connector):
    def __init__(self, reply_generator: DiscordReplyGenerator, connectors_event: Event):
        Connector.__init__(self, reply_generator=reply_generator, connectors_event=connectors_event)
        self._scheduler = ZmqScheduler(self._shutdown_event)