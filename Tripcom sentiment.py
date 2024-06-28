import pandas as pd
from textblob import TextBlob
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# Load the CSV file
data = pd.read_csv('src/Tripcom_English.csv')

# Function to get the sentiment of a review
def get_sentiment(text):
    blob = TextBlob(text)
    return blob.sentiment.polarity

# Apply the sentiment analysis on the 'content' column
data['sentiment_score'] = data['content'].apply(get_sentiment)

# Categorize the sentiment based on the sentiment score
def categorize_sentiment(score):
    if score > 0:
        return 'positive'
    elif score < 0:
        return 'negative'
    else:
        return 'neutral'

data['sentiment_label'] = data['sentiment_score'].apply(categorize_sentiment)

# Convert the 'time' column to datetime format if not already done
data['time'] = pd.to_datetime(data['time'], format='%d/%m/%y')

# Correlation Analysis
corr = data[['rating', 'sentiment_score', 'image']].corr()
plt.figure(figsize=(10, 8))
sns.heatmap(corr, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.title('Correlation Heatmap')
plt.show()

# N-gram Analysis
def get_top_n_grams(corpus, n=None, ngram_range=(1, 1)):
    vec = CountVectorizer(ngram_range=ngram_range, stop_words='english').fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0)
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq = sorted(words_freq, key=lambda x: x[1], reverse=True)
    return words_freq[:n]

top_unigrams = get_top_n_grams(data['content'], n=10, ngram_range=(1, 1))
top_bigrams = get_top_n_grams(data['content'], n=10, ngram_range=(2, 2))

# Plot top unigrams
unigrams_df = pd.DataFrame(top_unigrams, columns=['word', 'freq'])
plt.figure(figsize=(10, 6))
sns.barplot(x='freq', y='word', data=unigrams_df, hue='word', palette='viridis', legend=False)
plt.title('Top 10 Unigrams')
plt.xlabel('Frequency')
plt.ylabel('Unigrams')
plt.show()

# Plot top bigrams
bigrams_df = pd.DataFrame(top_bigrams, columns=['word', 'freq'])
plt.figure(figsize=(10, 6))
sns.barplot(x='freq', y='word', data=bigrams_df, hue='word', palette='viridis', legend=False)
plt.title('Top 10 Bigrams')
plt.xlabel('Frequency')
plt.ylabel('Bigrams')
plt.show()

# Topic Modeling with LDA
vectorizer = CountVectorizer(stop_words='english')
X = vectorizer.fit_transform(data['content'])

lda = LatentDirichletAllocation(n_components=5, random_state=42)
lda.fit(X)

feature_names = vectorizer.get_feature_names_out()
topics = []
for idx, topic in enumerate(lda.components_):
    topics.append([feature_names[i] for i in topic.argsort()[-10:]])

# Plot topics
fig, axes = plt.subplots(1, 5, figsize=(15, 5), sharex=True)
axes = axes.flatten()

for topic_idx, topic in enumerate(topics):
    top_features = topic
    weights = lda.components_[topic_idx]
    top_weights = np.array(weights)[np.argsort(weights)[-10:]]

    ax = axes[topic_idx]
    ax.barh(top_features, top_weights, color='blue')
    ax.set_title(f'Topic {topic_idx + 1}')
    ax.set_xlabel('Weight')
    ax.set_ylabel('Feature')

plt.tight_layout()
plt.show()

# Temporal Sentiment Analysis for a Specific Attraction
attraction_name = 'Botanic Garden'  # Replace with an actual attraction name
attraction_data = data[data['Attraction'] == attraction_name]

if not attraction_data.empty:
    plt.figure(figsize=(12, 6))
    attraction_data.groupby(attraction_data['time'].dt.to_period('M'))['sentiment_score'].mean().plot(kind='line')
    plt.title(f'Sentiment Score Over Time for {attraction_name}')
    plt.xlabel('Time')
    plt.ylabel('Average Sentiment Score')
    plt.show()
else:
    print(f"No data available for {attraction_name}")

# Plot sentiment distribution for each attraction
plt.figure(figsize=(15, 10))
sns.boxplot(x='Attraction', y='sentiment_score', data=data)
plt.xticks(rotation=90)
plt.title('Sentiment Distribution by Attraction')
plt.xlabel('Attraction')
plt.ylabel('Sentiment Score')
plt.show()

# Top positive reviews
top_positive_reviews = data.nlargest(10, 'sentiment_score')
print("Top 10 Positive Reviews:\n", top_positive_reviews[['content', 'sentiment_score']])

# Top negative reviews
top_negative_reviews = data.nsmallest(10, 'sentiment_score')
print("\nTop 10 Negative Reviews:\n", top_negative_reviews[['content', 'sentiment_score']])


# Preprocessing and splitting the data
X = data['content']
y = data['sentiment_label']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Vectorize the text data using Bag of Words
vectorizer = CountVectorizer(stop_words='english')
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Train a Naive Bayes classifier
nb_classifier = MultinomialNB()
nb_classifier.fit(X_train_vec, y_train)

# Predict the labels for the test set
y_pred = nb_classifier.predict(X_test_vec)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy * 100:.2f}%')

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10, 7))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=nb_classifier.classes_, yticklabels=nb_classifier.classes_)
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

