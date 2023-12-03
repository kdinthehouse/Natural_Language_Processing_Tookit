from nltk import FreqDist
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.probability import FreqDist

def word_frequency(text):
    words = word_tokenize(text)
    fdist = FreqDist(words)
    return fdist
