import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
from spellchecker import SpellChecker

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

def remove_stopwords(text):
    stop_words = set(stopwords.words('english'))
    words = word_tokenize(text)
    filtered_text = [word for word in words if word.lower() not in stop_words]
    return ' '.join(filtered_text)

def stemming(text):
    ps = PorterStemmer()
    words = word_tokenize(text)
    stemmed_text = [ps.stem(word) for word in words]
    return ' '.join(stemmed_text)

def lemmatization(text):
    lemmatizer = WordNetLemmatizer()
    words = word_tokenize(text)
    lemmatized_text = [lemmatizer.lemmatize(word) for word in words]
    return ' '.join(lemmatized_text)

def spelling_correction(text):
    spell = SpellChecker()
    words = word_tokenize(text)
    corrected_text = [spell.correction(word) for word in words]
    return ' '.join(corrected_text)
