import numpy as np
import pandas as pd

import nltk
from nltk.corpus import wordnet
import gensim
from tqdm import tqdm

import unicodedata
import re
from textblob import Word
import spacy
from bs4 import BeautifulSoup

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans, AgglomerativeClustering
from collections import Counter

from ref_list import CONTRACTION_MAP_EN




nlp = spacy.load('en_core_web_sm')
stopword_list = nltk.corpus.stopwords.words('english')


# Function to set default display options for Pandas
def set_default_pandas_options(max_columns=10, max_rows=2000, width=1000, max_colwidth=50):
    pd.set_option('display.max_columns', max_columns)
    pd.set_option('display.max_rows', max_rows)
    pd.set_option('display.width', width)
    pd.set_option('max_colwidth', max_colwidth)


# Function to write the given HTML content to the specified file.
def write_html_to_file(filename, html): 
    f = open(filename, 'w')
    f.write(html)
    f.close()


# Function to generate data quality report   
def data_quality_report(df):
    if isinstance(df, pd.core.frame.DataFrame): 
        descriptive_statistics = df.describe(include='all')
        data_types = pd.DataFrame(df.dtypes, columns=['Data Type']).transpose()
        missing_value_counts = pd.DataFrame(df.isnull().sum(), columns=['Missing Values']).transpose()
        present_value_counts = pd.DataFrame(df.count(), columns=['Present Values']).transpose()
        data_report = pd.concat([descriptive_statistics, data_types, missing_value_counts, present_value_counts])
        return data_report
    else:
        return None



    
# Text Normalizer

# Function to split the text into sentences and return a list of sentences
def tokenize_text_to_sentences(text):
    sentences = nltk.sent_tokenize(text)
    return sentences


# Function to break the sentence into words and return a list of words
def tokenize_sentence_to_words(sentence):
    words = nltk.word_tokenize(sentence)
    return words


# Function to remove the HTML tag from the text and return plain text
def strip_html_tags(text):
    # Parse HTML text using the BeautifulSoup library
    soup = BeautifulSoup(text, 'html.parser')
    
    if soup.find():
        [s.extract() for s in soup(['iframe', 'script'])] # Remove specific tags (iframe and script)
        stripped_text = soup.get_text() # Get plain text from parsed text
        stripped_text = re.sub(r'[\r|\n|\r\n]+', '\n', stripped_text) # Replace consecutive newlines
    else:
        stripped_text = text
    
    # Return plain text without the HTML tag
    return stripped_text


# Function to remove accented characters from text
def remove_accented_chars(text):
    text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore')
    return text


# Function to replace abbreviations with full phrases through the provided contraction_mapping
def expand_contractions(text, contraction_mapping=CONTRACTION_MAP_EN):
    sorted_contractions = sorted(contraction_mapping.keys(), key=len, reverse=True)
    contractions_pattern = re.compile(r'\b(?:' + '|'.join(re.escape(key) for key in sorted_contractions) + r')\b', flags=re.IGNORECASE)

    def expand_match(contraction):
        match = contraction.group(0)
        expanded_contraction = contraction_mapping.get(match.lower(), match)
        return expanded_contraction
    
    expanded_text = contractions_pattern.sub(expand_match, text)
    expanded_text = expanded_text.replace("'", "")
    
    return expanded_text


# Function to removes special characters from text
# Special characters are all characters other than letters, numbers and blank characters
# The "remove_digits" parameter controls whether to delete numbers
def remove_special_characters(text, remove_digits=False):
    pattern = r'[^a-zA-Z0-9\s]' if not remove_digits else r'[^a-zA-Z\s]'
    text = re.sub(pattern, '', text)
    text = text.replace('[', '').replace(']', '')
    return text


# Function to remove duplicate characters from words
def remove_repeated_characters(tokens): 
    repeat_pattern = re.compile(r'(\w*)(\w)\2(\w*)')
    match_substitution = r'\1\2\3'
    
    def replace(old_word):
        if wordnet.synsets(old_word):
            return old_word
        
        new_word = repeat_pattern.sub(match_substitution, old_word)
        return replace(new_word) if new_word != old_word else new_word
    
    correct_tokens = [replace(word) for word in tokens]
    
    return correct_tokens


# Function to correct spelling for a given list of words
def correct_spelling(word_tokens):
    corrected_tokens = [Word(word).correct() for word in word_tokens]
    return [str(token) for token in corrected_tokens]


# Function to lemmatizate a given list of words
def lemmatize_tokens(tokens):
    lemmatized_tokens = []
    
    for token in tokens:
        if token == '-PRON-':
            lemmatized_tokens.append(token)
        else:
            lemmatized_tokens.append(nlp(token)[0].lemma_)

    return lemmatized_tokens


# Function to remove the stopword from the given list of words
def remove_stopword(tokens, is_lower_case=False):
    stopword_set = set(stopword_list)
    return ['' if (is_lower_case and token in stopword_set) or (not is_lower_case and token.lower() in stopword_set) else token for token in tokens]


# Function to exclude some stopwords from the stopword_list
def exclude_stopwords(stopword_exclusion_list):
    for exclude in stopword_exclusion_list: 
        stopword_list.remove(exclude)


# Function to clean raw text applying above functions
def normalize_corpus(dataframe, raw_column, clean_column,
                     html_stripping=False,
                     accented_char_removal=True, contraction_expansion=True,
                     text_lower_case=True, extra_newlines_removal=True, extra_whitespace_removal=True,
                     special_char_removal=True, remove_digits=True, repeating_char_removal=True,
                     spelling_correction=True, lemmatize=True, stop_word_removal=True):
    
    def process_row(row):
        text = row[raw_column]

        if html_stripping:
            text = strip_html_tags(text)
        if accented_char_removal:
            text = remove_accented_chars(text)
        if contraction_expansion:
            text = expand_contractions(text)
        if text_lower_case:
            text = text.lower()
        if extra_newlines_removal:
            text = re.sub(r'[\r|\n|\r\n]+', ' ', text)
        if extra_whitespace_removal:
            text = re.sub(' +', ' ', text)
        if special_char_removal:
            text = remove_special_characters(text, remove_digits)

        # tokenize into words
        word_tokens = tokenize_sentence_to_words(text)

        if repeating_char_removal:
            word_tokens = remove_repeated_characters(word_tokens)
        if spelling_correction:
            word_tokens = correct_spelling(word_tokens)
        if lemmatize:
            word_tokens = lemmatize_tokens(word_tokens)
        if stop_word_removal:
            word_tokens = remove_stopword(word_tokens, text_lower_case)

        word_tokens = [word_token for word_token in word_tokens if word_token != '']
        cleaned_text = ' '.join(word_tokens)

        return cleaned_text

    dataframe[clean_column] = dataframe.apply(process_row, axis=1)

    return dataframe




# Keywords Extraction

# Function to count word frequency
def get_word_frequency(texts):
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(texts)
    word_freq = pd.DataFrame(X.toarray(), columns=vectorizer.get_feature_names_out())
    return word_freq


# Function to extract TF-IDF keyword
def get_tfidf_keywords(texts, n):
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(texts)
    feature_names = vectorizer.get_feature_names_out()
    tfidf_df = pd.DataFrame(X.toarray(), columns=feature_names)
    keywords = tfidf_df.sum().sort_values(ascending=False)[:n]  # Extract the top n keywords
    return keywords


# Function to extract key phrases
def extract_phrases(texts, n_words):
    vectorizer = TfidfVectorizer(ngram_range=(n_words, n_words))
    X = vectorizer.fit_transform(texts)
    phrases = vectorizer.get_feature_names_out()
    return phrases

def extract_phrases(texts, n_words):
    vectorizer = TfidfVectorizer(ngram_range=(n_words, n_words))
    X = vectorizer.fit_transform(texts)
    phrases = vectorizer.get_feature_names_out()
    
    # Sort by frequency in descending order
    phrase_counts = Counter(phrases)
    sorted_phrases = sorted(phrase_counts.items(), key=lambda x: x[1], reverse=True)
    sorted_phrases = [phrase for phrase, count in sorted_phrases]
    return sorted_phrases




# Similarity Matching

# Function to clustering by cosine_similarity
def cluster_similar_texts(texts, threshold):
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(texts)

    # Calculate similarity matrix
    similarity_matrix = cosine_similarity(X)

    # Classify
    classified_texts = {}
    for i in range(len(texts)):
        if i not in classified_texts:
            similar_indices = [j for j, similarity_score in enumerate(similarity_matrix[i]) if similarity_score > threshold]
            similar_texts = [texts[j] for j in similar_indices]
            classified_texts[i] = similar_texts

    return classified_texts




# Clustering Analysis

# Function to Kmeans clustering
def cluster_kmeans(texts, num_clusters):
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(texts)
    
    kmeans = KMeans(n_clusters=num_clusters)
    kmeans.fit(X)
    
    return kmeans.labels_


# Function to hierarchical clustering
def cluster_hierarchical(texts, num_clusters):
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(texts)
    
    hierarchical_clustering = AgglomerativeClustering(n_clusters=num_clusters)
    cluster_labels = hierarchical_clustering.fit_predict(X.toarray())  # Convert sparse matrix to dense array
    
    return cluster_labels




# Word2vec Embeddings

# Function to calculate the average word vector for a given set of words
def average_word_vectors(words, model, vocabulary, num_features):
    valid_words = [word for word in words if word in vocabulary]
    
    if valid_words:
        word_vectors = model.wv[valid_words]
        feature_vector = np.mean(word_vectors, axis=0)
    else:
        feature_vector = np.zeros((num_features,), dtype="float64")
    
    return feature_vector


# Function to convert each document (sentence or other text unit) 
# in the entire corpus into an average word vector representation
def averaged_word_vectorizer(corpus, model, num_features):
    vocabulary = set(model.wv.index_to_key)
    valid_corpuses = [[word for word in tokenized_sentence if word in vocabulary] for tokenized_sentence in corpus]
    features = [np.mean(model.wv[valid_words], axis=0) if valid_words else np.zeros((num_features,), dtype="float64")
                for valid_words in valid_corpuses]
    
    return np.array(features)




# Topic Modelling

# Function to generate topic models and calculate consistency scores for various topic models
def topic_model_coherence_generator(model_name,
                                    corpus, texts, dictionary, 
                                    start_topic_count=2, end_topic_count=3, step=1, cpus=1,
                                    print_topics=False):
    
    models = []
    coherence_scores = []

    for topic_nums in tqdm(range(start_topic_count, end_topic_count+1, step)):
        print('\nMODEL: {} - NUMBER OF TOPICS: {}'.format(model_name, topic_nums))
        
        if model_name == 'LSI':
            model = gensim.models.LsiModel(corpus=corpus,
                                           id2word=dictionary,
                                           num_topics=topic_nums,
                                           chunksize=1740,
                                           power_iters=1000)
        
        elif model_name == 'LDA':
            model = gensim.models.LdaModel(corpus=corpus, 
                                           id2word=dictionary,
                                           num_topics=topic_nums,
                                           chunksize=1740,
                                           alpha='auto',
                                           eta='auto',
                                           iterations=500)
        
        cv_coherence_model = gensim.models.CoherenceModel(model=model,
                                                          corpus=corpus,
                                                          texts=texts,
                                                          dictionary=dictionary,
                                                          coherence='c_v',
                                                          processes=cpus)
        
        
        coherence_score = cv_coherence_model.get_coherence()
        
        coherence_scores.append(coherence_score)
        models.append(model)
        
        if print_topics:
            for topic_id, topic in model.print_topics(num_topics=10, num_words=20):
                print('Topic #'+str(topic_id+1)+':')
                print('='*10)
                print(topic)
                print()
                
            print('-'*50)

    return models, coherence_scores



# Define Formatting

# Define for bold and remove bold
bold = "\033[1m"
reset = "\033[0m"

# Function to check NULL values in dataframe
def check_null(df, df_name):
    print(f"{bold}{df_name}:{reset}")
    print(df.isnull().sum())
    print(f"{bold}Total rows in {df_name}:{reset}", len(df), "\n")