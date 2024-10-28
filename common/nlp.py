from typing import Optional, List, Tuple
from enum import Enum, unique
from common.ml import one_hot, MLDataPreprocessor
import torch
import re
import spacy
import re
from spacy.util import filter_spans
from spacymoji import Emoji
from spacy.tokens import Token, Doc
from spacy.matcher import Matcher
from config.nlp_config import USE_GPU, USE_TRANSFORMER

# Precompile regex patterns
DISCORD_PATTERNS = {
    "USER_MENTION": re.compile(r"<@!?(\d+)>"),
    "CHANNEL_MENTION": re.compile(r"<#(\d+)>"),
    "ROLE_MENTION": re.compile(r"<@&(\d+)>"),
    "EMOJI_MENTION": re.compile(r"<a?:\w+:\d+>"),
    "URL": re.compile(r'https?://[^\s]+'),
    "SPOILER": re.compile(r"\|\|(.*?)\|\|"),
    "ATTACHMENT": re.compile(r"https?://cdn\.discordapp\.com/attachments/\d+/\d+/[^\s]+"),
    "BOT_COMMAND": re.compile(r'^[!/]')
}

MARKDOWN_PATTERNS = {
    "BOLD": re.compile(r"\*\*(.*?)\*\*"),
    "ITALIC": re.compile(r"(\*|_)(.*?)\1"),
    "STRIKETHROUGH": re.compile(r"~~(.*?)~~"),
    "CODE": re.compile(r"`([^`]+)`"),
    "CODE_BLOCK": re.compile(r"```([\s\S]*?)```")
}

def create_nlp_instance():
    if USE_GPU:
        try:
            spacy.require_gpu()
        except:
            pass
    
    nlp = None

    if USE_TRANSFORMER:
        nlp = spacy.load('en_core_web_trf')
    else:
        nlp = spacy.load('en_core_web_lg')

    nlp.add_pipe("emoji", before="parser")

    # Initialize a Matcher
    matcher = Matcher(nlp.vocab)

    # Add patterns to the matcher
    for label, pattern in DISCORD_PATTERNS.items():
        matcher.add(label, [[{"TEXT": {"REGEX": pattern.pattern}}]])

    for label, pattern in MARKDOWN_PATTERNS.items():
        matcher.add(label, [[{"TEXT": {"REGEX": pattern.pattern}}]])

    @nlp.component("combined_discord_patterns")
    def combined_discord_patterns(doc):
        matches = matcher(doc)
        spans = []
        for match_id, start, end in matches:
            label = nlp.vocab.strings[match_id]
            span = doc[start:end]
            spans.append(spacy.tokens.Span(doc, start, end, label=label))
        
        # Filter overlapping spans
        spans = filter_spans(spans)

        if spans:
            with doc.retokenize() as retokenizer:
                for span in spans:
                    retokenizer.merge(span)
        return doc
    nlp.add_pipe("combined_discord_patterns", before="parser")

    @nlp.component("remove_trf_data")
    def remove_trf_data(doc):
        doc._.trf_data = None
        return doc
    nlp.add_pipe("remove_trf_data")

    return nlp

@unique
class Pos(Enum):
    NONE = 0

    # Universal POS tags
    ADJ = 1
    ADP = 2
    ADV = 3
    AUX = 4
    CONJ = 5
    CCONJ = 6
    DET = 7
    INTJ = 8
    NOUN = 9
    NUM = 10
    PART = 11
    PRON = 12
    PROPN = 13
    PUNCT = 14
    SCONJ = 15
    SYM = 16
    VERB = 17
    X = 18
    SPACE = 19

    # Custom POS tags
    EMOJI = 20
    HASHTAG = 21
    URL = 22
    USER_MENTION = 23
    CHANNEL_MENTION = 24
    ROLE_MENTION = 25
    EMOJI_MENTION = 26
    BOLD = 27
    ITALIC = 28
    STRIKETHROUGH = 29
    CODE = 30
    CODE_BLOCK = 31
    SPOILER = 32
    ATTACHMENT = 33
    BOT_COMMAND = 34

    # Special
    EOS = 35

    def one_hot(self) -> list:
        return one_hot(self.value, len(Pos))

    @staticmethod
    def from_token(token: spacy.tokens.Token, people: list = None) -> Optional['Pos']:
        ENTITY_POS_MAPPING = {
            "USER_MENTION": Pos.USER_MENTION,
            "CHANNEL_MENTION": Pos.CHANNEL_MENTION,
            "ROLE_MENTION": Pos.ROLE_MENTION,
            "EMOJI_MENTION": Pos.EMOJI_MENTION,
            "URL": Pos.URL,
            "SPOILER": Pos.SPOILER,
            "ATTACHMENT": Pos.ATTACHMENT,
            "BOT_COMMAND": Pos.BOT_COMMAND,
            "BOLD": Pos.BOLD,
            "ITALIC": Pos.ITALIC,
            "STRIKETHROUGH": Pos.STRIKETHROUGH,
            "CODE": Pos.CODE,
            "CODE_BLOCK": Pos.CODE_BLOCK
        }

        if token.ent_type_ in ENTITY_POS_MAPPING:
            print(token.ent_type_)
            return ENTITY_POS_MAPPING[token.ent_type_]
        
        if token.text.startswith('@'):
            return Pos.PROPN
        if token.text.startswith(' ') or token.text == "\n":
            return Pos.SPACE
        if token._.is_emoji:
            return Pos.EMOJI

        if people is not None:
            if token.text in people:
                return Pos.PROPN

        if re.match(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', token.text):
            return Pos.URL
        if re.match(r'<[&@]\d{17,19}>', token.text):
            return Pos.PROPN
        if re.match(r'<:[^:]+:\d+>', token.text):
            return Pos.EMOJI
        if re.match(r'<#\d{17,19}>', token.text):
            return Pos.PROPN
        if re.match(r'[?~!\/.$][a-zA-Z]+', token.text):
            return Pos.NOUN

        try:
            return Pos[token.pos_]
        except KeyError:
            print(f"Unknown PoS: {token.text}")
            return Pos.X


@unique
class CapitalizationMode(Enum):
    NONE = 0
    UPPER_FIRST = 1
    UPPER_ALL = 2
    LOWER_ALL = 3
    COMPOUND = 4

    def one_hot(self):
        ret_list = []
        for i in range(0, len(CapitalizationMode)):
            if i != self.value:
                ret_list.append(0)
            else:
                ret_list.append(1)
        return ret_list

    @staticmethod
    def from_token(token: Token, compound_rules: Optional[List[str]] = None) -> 'CapitalizationMode':

        # Try to make a guess for many common patterns
        pos = Pos.from_token(token)
        if pos in [Pos.NUM, Pos.EMOJI, Pos.SYM, Pos.SPACE, Pos.EOS, Pos.HASHTAG, Pos.PUNCT, Pos.URL]:
            return CapitalizationMode.COMPOUND

        if token.text[0] == '@' or token.text[0] == '#':
            return CapitalizationMode.COMPOUND

        if token.text in compound_rules:
            return CapitalizationMode.COMPOUND

        lower_count = 0
        upper_count = 0
        upper_start = False
        for idx, c in enumerate(token.text):

            if c.isupper():
                upper_count += 1
            if upper_start:
                upper_start = False
            if idx == 0:
                upper_start = True
            elif c.islower():
                lower_count += 1

        if upper_start:
            return CapitalizationMode.UPPER_FIRST
        elif lower_count > 0 and upper_count == 0:
            return CapitalizationMode.LOWER_ALL
        elif upper_count > 0 and lower_count == 0:
            return CapitalizationMode.UPPER_ALL
        elif upper_count == 0 and lower_count == 0:
            return CapitalizationMode.NONE
        else:
            return CapitalizationMode.COMPOUND

    @staticmethod
    def transform(mode: 'CapitalizationMode', word: str) -> str:

        ret_word = word

        if mode == CapitalizationMode.UPPER_FIRST:

            first_alpha_flag = False
            ret_list = []

            ret_word = ret_word.lower()

            # Find the first letter
            for c_idx, c in enumerate(ret_word):
                if c.isalpha() and not first_alpha_flag:
                    ret_list.append(ret_word[c_idx].upper())
                    first_alpha_flag = True
                else:
                    ret_list.append(c)
            ret_word = "".join(ret_list)

        elif mode == CapitalizationMode.UPPER_ALL:
            ret_word = ret_word.upper()
        elif mode == CapitalizationMode.LOWER_ALL:
            ret_word = ret_word.lower()

        return ret_word

class SpacyPreprocessor(MLDataPreprocessor):
    def __init__(self):
        MLDataPreprocessor.__init__(self, 'SpacyPreprocessor')

    def preprocess(self, doc: spacy.tokens.Doc) -> bool:
        self.data.append(doc)
        return True

    def get_preprocessed_data(self) -> Tuple[List, List]:
        return self.data, self.labels
