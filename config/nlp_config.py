# --- "User" Stuff Section ---
# ----------------------------

USE_GPU = True

# --- Technical Stuff Section ---
# -------------------------------

# Paths
MARKOV_DB_PATH = 'weights/markov'
REACTION_MODEL_PATH = "weights/aol-reaction-model.h5"
STRUCTURE_MODEL_PATH = "weights/pytorch_model.pth" #"weights/structure-model.weights.h5"

MARKOV_GENERATE_SUBJECT_MAX = 1000000
# Greatest to least
MARKOV_GENERATE_SUBJECT_POS_PRIORITY = [
    "BOT_COMMAND",       # Highly relevant for bot interaction
    "PROPN",             # Proper nouns for key entities
    "NOUN",              # Nouns for subjects and objects
    "VERB",              # Verbs to represent actions
    "ADJ",               # Adjectives for descriptive content
    "ADV",               # Adverbs for additional details
    "EMOJI",             # Emojis for expressive elements
    "INTJ",              # Interjections for exclamations or emotion
    "NUM",               # Numbers for quantities
    "HASHTAG",           # Hashtags for trending topics
    "URL",               # URLs for references or links
    "X",                 # Unknown or other tokens
    "SYM",               # Symbols for special characters
    "PART",              # Particles for grammatical structure
    "DET",               # Determiners for specificity
    "AUX",               # Auxiliary verbs for tense/aspect
    "CONJ",              # Conjunctions for connecting ideas
    "CCONJ",             # Coordinating conjunctions (e.g., "and", "or")
    "SCONJ",             # Subordinating conjunctions (e.g., "because", "although")
    "PRON",              # Pronouns for references to entities
    "ADP",               # Adpositions like prepositions
    "BOLD",              # Bold text for emphasis
    "ITALIC",            # Italics for subtle emphasis
    "STRIKETHROUGH",     # Strikethrough for indicating removal or correction
    "CODE",              # Code snippets for technical content
    "CODE_BLOCK",        # Larger code blocks
    "SPOILER",           # Spoiler content hidden by default
    "USER_MENTION",      # Mentions of specific users
    "CHANNEL_MENTION",   # Mentions of specific channels
    "ROLE_MENTION",      # Mentions of user roles
    "EMOJI_MENTION",     # Mentions specifically using emojis
    "ATTACHMENT",        # Attachments with additional content
    "PUNCT",             # Punctuation for structure
    "SPACE",             # Spaces for formatting
    "EOS",               # End of sentence marker
    "NONE"               # None, for cases with no specific POS
]
# XDD
MAX_SEQUENCE_LEN = 16


# Weights for generating replies Make u decide if u prefers frequency or importante his
MARKOV_GENERATION_WEIGHT_COUNT = 8
MARKOV_GENERATION_WEIGHT_RATING = 2

# bi-gram window function size
MARKOV_WINDOW_SIZE = 6

# These should always be marked as a "compound" word which will always use its original capitalization
CAPITALIZATION_COMPOUND_RULES = ['RT']

# Maximum number of sequences to train the structure model on
STRUCTURE_MODEL_TRAINING_MAX_SIZE = 20000000
MARKOV_MODEL_MAX_PREPROCESSING_SIZE = 250000
# Lower values make things more predictable, higher ones more random
STRUCTURE_MODEL_TEMPERATURE = 0.7
MARKOV_MODEL_TEMPERATURE = 0.5

#tekob specially character
FILTER_SPECIALLY = False

#use transformerka model for nlp POS tagging (goods)
USE_TRANSFORMER = True
