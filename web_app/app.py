from flask import Flask, render_template, request
from nltk_tools.text_preprocessing import remove_stopwords, lemmatization
from nltk_tools.sentiment_analysis import sentiment_analysis
from nltk_tools.topic_modeling import word_frequency
from spacy_tools.text_preprocessing import remove_stopwords_spacy
from spacy_tools.sentiment_analysis import sentiment_analysis_spacy
from spacy_tools.topic_modeling import topic_modeling_spacy
from gensim_tools.word_vector_analysis import word_vector_analysis
from gensim_tools.clustering import word_cluster_analysis

app = Flask(__name__)

# Sample sentences for word vector analysis
sample_sentences = [
    "Natural Language Processing is a subfield of artificial intelligence.",
    "Machine learning algorithms can analyze and understand human language.",
    "NLTK and spaCy are popular libraries for NLP tasks.",
    "Word embeddings capture semantic relationships between words.",
    "Clustering groups similar words together based on their vector representations."
]

# Load word vector model
word_vector_model = word_vector_analysis(sample_sentences)

def run_custom_pipeline(user_input, lang='en'):
    # Perform NLP tasks using NLTK
    filtered_text_nltk = remove_stopwords(user_input)
    lemmatized_text_nltk = lemmatization(user_input)
    sentiment_score_nltk = sentiment_analysis(user_input)
    freq_dist_nltk = word_frequency(user_input)

    # Perform NLP tasks using spaCy
    filtered_text_spacy = remove_stopwords_spacy(user_input, lang=lang)
    sentiment_score_spacy = sentiment_analysis_spacy(user_input, lang=lang)
    topics_spacy = topic_modeling_spacy(user_input, lang=lang)

    # Perform word vector analysis
    word_clusters = word_cluster_analysis(word_vector_model, num_clusters=3)

    return {
        'filtered_text_nltk': filtered_text_nltk,
        'lemmatized_text_nltk': lemmatized_text_nltk,
        'sentiment_score_nltk': sentiment_score_nltk,
        'freq_dist_nltk': freq_dist_nltk,
        'filtered_text_spacy': filtered_text_spacy,
        'sentiment_score_spacy': sentiment_score_spacy,
        'topics_spacy': topics_spacy,
        'word_clusters': word_clusters
    }

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Get user input
        user_input = request.form['user_input']
        lang = request.form['lang']

        # Run custom pipeline
        result = run_custom_pipeline(user_input, lang=lang)

        # Render the result on the webpage
        return render_template('index.html', user_input=user_input, lang=lang, result=result)

    # Render the main webpage
    return render_template('index.html', user_input=None, lang='en', result=None)

if __name__ == '__main__':
    app.run(debug=True)