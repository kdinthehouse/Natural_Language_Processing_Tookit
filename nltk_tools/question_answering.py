from nltk import pos_tag
from nltk.chunk import ne_chunk
from nltk.tokenize import word_tokenize

def extract_entities(text):
    words = word_tokenize(text)
    pos_tags = pos_tag(words)
    named_entities = ne_chunk(pos_tags)
    return named_entities
