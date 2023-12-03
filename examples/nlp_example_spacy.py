from spacy_tools.text_preprocessing import remove_stopwords_spacy
from spacy_tools.named_entity_recognition import named_entity_recognition_spacy
from spacy_tools.dependency_parsing import dependency_parsing
from spacy_tools.sentiment_analysis import sentiment_analysis_spacy
from spacy_tools.topic_modeling import topic_modeling_spacy

text = "Natural Language Processing is a subfield of artificial intelligence."

# Example usage
filtered_text_spacy = remove_stopwords_spacy(text, lang='en')
entities_spacy = named_entity_recognition_spacy(text, lang='en')
dependencies_spacy = dependency_parsing(text, lang='en')
sentiment_score_spacy = sentiment_analysis_spacy(text, lang='en')
topics_spacy = topic_modeling_spacy(text, lang='en')

print("Filtered Text (spaCy):", filtered_text_spacy)
print("Named Entities (spaCy):", entities_spacy)
print("Dependency Parsing (spaCy):", dependencies_spacy)
print("Sentiment Score (spaCy):", sentiment_score_spacy)
print("Topics (spaCy):", topics_spacy)
