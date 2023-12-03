import spacy

def remove_stopwords_spacy(text, lang='en'):
    nlp = spacy.load(f'{lang}_core_web_sm')
    doc = nlp(text)
    tokens = [token.text for token in doc if not token.is_stop]
    return ' '.join(tokens)
