from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import pandas as pd


data = pd.read_csv('ctrip/Ctrip_clean_review.csv', encoding='utf_8_sig', index_col=False)

# Prepare the text data
cleaned_content = data['cleaned_content'].dropna().tolist()

# Use CountVectorizer to convert the text data to a matrix of token counts
vectorizer = CountVectorizer()
text_data = vectorizer.fit_transform(cleaned_content)

# Perform topic modeling using LDA
num_topics = 5  # You can adjust the number of topics
lda = LatentDirichletAllocation(n_components=num_topics, random_state=42)
lda.fit(text_data)

# Get the topics and their top words
def display_topics(model, feature_names, num_top_words):
    topics = []
    for topic_idx, topic in enumerate(model.components_):
        topics.append([feature_names[i] for i in topic.argsort()[:-num_top_words - 1:-1]])
    return topics

num_top_words = 10
feature_names = vectorizer.get_feature_names_out()
topics = display_topics(lda, feature_names, num_top_words)

print(topics)