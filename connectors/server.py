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
from config.nlp_config import MARKOV_MODEL_TEMPERATURE, STRUCTURE_MODEL_TEMPERATURE
logging.basicConfig(
    level=MARKOV_SERVER_LOG_LEVEL,
    format='[%(asctime)s][%(levelname)s][%(name)s]: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("ZMQ SERVER")
FALLBACK_CONFIG = {
    'strategy': 'softmax',  # Default strategy if invalid
    'temperature': MARKOV_MODEL_TEMPERATURE,     # Default temperature
    'top_k': 50,            # Default top_k for Top-k
    'top_p': 0.9,           # Default top_p for Top-p
}

def validate_sampling_config(sampling_config):
    global FALLBACK_CONFIG

    # Check if the necessary keys are present and not None
    required_keys = ['strategy', 'temperature', 'top_k', 'top_p']
    
    for key in required_keys:
        if key not in sampling_config or sampling_config[key] is None:
            logger.info(f"Warning: '{key}' is missing or None. Using default values.")
            sampling_config[key] = FALLBACK_CONFIG[key]

    # If 'strategy' is invalid, set it to the default value
    if sampling_config['strategy'] not in ['softmax', 'top_p', 'top_k', 'greedy', 'random', 'top_p_k']:
        logger.warning(f"Warning: Invalid strategy '{sampling_config['strategy']}'. Using default 'softmax'.")
        sampling_config['strategy'] = FALLBACK_CONFIG['strategy']

    # Ensure numerical values are valid
    if not isinstance(sampling_config['temperature'], (int, float)) or sampling_config['temperature'] <= 0.09:
        logger.warning("Warning: Invalid temperature. Setting to default 1.0.")
        sampling_config['temperature'] = FALLBACK_CONFIG['temperature']
    
    if not isinstance(sampling_config['top_k'], int) or sampling_config['top_k'] <= 0:
        logger.warning("Warning: Invalid top_k value. Setting to default 50.")
        sampling_config['top_k'] = FALLBACK_CONFIG['top_k']
    
    if not isinstance(sampling_config['top_p'], (int, float)) or not (0 < sampling_config['top_p'] <= 1):
        logger.warning("Warning: Invalid top_p value. Setting to default 0.9.")
        sampling_config['top_p'] = FALLBACK_CONFIG['top_p']
    
    return sampling_config

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
    
    def generate(self, message: str, sampling_config: dict, doc: Doc=None, ignore_topics = None) -> Optional[str]:
        reply = super().generate(message, sampling_config, doc, ignore_topics)
        
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

    async def handle_message(self, message):
        try:
              # raw=False to get strings as UTF-8
            if message['type'] == 'ping':
                return ({'type': 'pong', 'from': 'markov'})
            logger.info(message)
            if not message.get('text', ""):
                return None
            if not message.get('sampling_config', {}):
                message['sampling_config'] = {'markov': {}, 'struct': {}}
            message['sampling_config']['markov'] = validate_sampling_config(message['sampling_config']['markov'])
            message['sampling_config']['struct'] = validate_sampling_config(message['sampling_config']['struct'])
            logger.info(message['sampling_config'])
            huharraq = ConnectorRecvMessage(
                message['text'],
                message['learn'], 
                message['reply'],
                message['sampling_config'], 
                message.get('ignore_topics', []),
            )
            self.send(huharraq)
            reply = self.recv() 
            if message['store'] and message['dmessage']:
                self._db.store_dict(message['dmessage'])  # Store the deserialized object
            return reply
        except Exception as e:
            logger.error(e)
            return "HUUNGNANHJANFHAISFJ ASNFUASH WASGW FAM WHA WKD"

    async def zmq_server_(self):
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
                
    async def zmq_server(self):
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REP)
        self.socket.bind(f"tcp://127.0.0.1:{MARKOV_PORT}")
        print(f"SERCER PORT {MARKOV_PORT}")
        while not self._shutdown_event.is_set():
            data = None
            try:
                message = self.socket.recv()
                data = None
                msg_type = None
                
                try:
                    data = msgpack.unpackb(message)
                    msg_type = 0
                except:
                    try:
                        data = json.loads(message.decode('utf-8'))
                        msg_type = 1
                    except:
                        logger.error("Failed to decode message as msgpack or JSON")
                        continue
                    
                response = await self.handle_message(data)
                if msg_type == 0:
                    packed_response = msgpack.packb(response)
                    self.socket.send(packed_response)#, flags=zmq.NOBLOCK)
                else:
                    packed_response = json.dumps(response)
                    self.socket.send_string(packed_response)#, flags=zmq.NOBLOCK)
            except Exception as e:
                if "current state" in str(e):
                    self.socket.close()
                    self.socket = self.context.socket(zmq.REP)
                    self.socket.bind(f"tcp://127.0.0.1:{MARKOV_PORT}")
                logger.error(e)
                logger.error(f"requesttt {data}")
                
    def run(self):
        #self._context = zmq.Context()
        #self._socket = self._context.socket(zmq.REP)
        from storage.message_storage import DiscordTrainingDataManager
        self._db = DiscordTrainingDataManager()

        #self._socket.bind(f"tcp://127.0.0.1:{MARKOV_PORT}")
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