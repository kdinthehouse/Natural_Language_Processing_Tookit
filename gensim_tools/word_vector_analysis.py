from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize

def word_vector_analysis(sentences):
    tokenized_sentences = [word_tokenize(sentence.lower()) for sentence in sentences]
    model = Word2Vec(tokenized_sentences, min_count=1)
    return model