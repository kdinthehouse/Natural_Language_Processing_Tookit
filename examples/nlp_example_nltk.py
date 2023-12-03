from nltk_tools.text_preprocessing import remove_stopwords, stemming, lemmatization, spelling_correction
from nltk_tools.sentiment_analysis import sentiment_analysis
from nltk_tools.topic_modeling import word_frequency
from nltk_tools.question_answering import extract_entities

text = "Natural Language Processing is a subfield of artificial intelligence."

# Example usage
filtered_text = remove_stopwords(text)
stemmed_text = stemming(text)
lemmatized_text = lemmatization(text)
corrected_text = spelling_correction(text)
sentiment_score = sentiment_analysis(text)
freq_dist = word_frequency(text)
entities = extract_entities(text)

print("Filtered Text:", filtered_text)
print("Stemmed Text:", stemmed_text)
print("Lemmatized Text:", lemmatized_text)
print("Spelling Corrected Text:", corrected_text)
print("Sentiment Score:", sentiment_score)
print("Word Frequency:", freq_dist)
print("Named Entities:", entities)
