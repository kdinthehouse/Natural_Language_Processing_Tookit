import spacy

def named_entity_recognition_spacy(text, lang='en'):
    nlp = spacy.load(f'{lang}_core_web_sm')
    doc = nlp(text)
    return [(ent.text, ent.label_) for ent in doc.ents]
