import spacy

def sentiment_analysis_spacy(text, lang='en'):
    nlp = spacy.load(f'{lang}_core_web_sm')
    doc = nlp(text)
    sentiment_score = doc.sentiment
    return sentiment_score