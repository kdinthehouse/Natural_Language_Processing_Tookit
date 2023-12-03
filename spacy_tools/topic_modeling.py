import spacy

def topic_modeling_spacy(text, lang='en'):
    nlp = spacy.load(f'{lang}_core_web_sm')
    doc = nlp(text)
    topics = [chunk.text for chunk in doc.noun_chunks]
    return topics