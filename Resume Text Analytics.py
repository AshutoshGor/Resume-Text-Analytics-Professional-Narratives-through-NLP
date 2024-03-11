#!/usr/bin/env python
# coding: utf-8

import math
import re
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
from gensim import corpora, models
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.util import ngrams
from wordcloud import WordCloud

# Load the CSV dataset
data = pd.read_csv("resumes.csv", encoding='iso-8859-1')
stop_words = set(stopwords.words('english'))

# Clean the text
def clean_text(text):
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Remove non-alphabetic characters
    words = word_tokenize(text)
    filtered_words = [word.lower() for word in words if word.lower() not in stop_words]
    cleaned_text = ' '.join(filtered_words)
    return cleaned_text

data['cleaned_text'] = data['resume_text'].apply(clean_text)

# Tokenize and generate bi-grams and tri-grams
def generate_ngrams(text, n):
    tokens = word_tokenize(text)
    n_grams = list(ngrams(tokens, n))
    return ["_".join(gram) for gram in n_grams]

data['bi_grams'] = data['cleaned_text'].apply(lambda x: generate_ngrams(x, 2))
data['tri_grams'] = data['cleaned_text'].apply(lambda x: generate_ngrams(x, 3))

# Prepare the data for LDA
texts = data['cleaned_text'].apply(word_tokenize)
dictionary = corpora.Dictionary(texts)
corpus = [dictionary.doc2bow(text) for text in texts]

# Build the LDA model
lda_model = models.LdaModel(corpus, num_topics=5, id2word=dictionary, passes=15)

# Get the topics
topics = lda_model.print_topics(num_words=5)
for topic in topics:
    print(topic)

# Generate a word cloud
wordcloud = WordCloud(width=800, height=400).generate(" ".join(data['cleaned_text']))

# Display the word cloud
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()

# Number of words
num_words = data['cleaned_text'].apply(lambda x: len(x.split())).sum()

# Total number of unique words
unique_words = set(" ".join(data['cleaned_text']).split())
num_unique_words = len(unique_words)

# Total entropy
word_counts = [data['cleaned_text'].apply(lambda x: x.split().count(word)).sum() for word in unique_words]
entropy = -sum((count / num_words) * math.log2(count / num_words) for count in word_counts)

print(f"Number of words: {num_words}")
print(f"Total number of unique words: {num_unique_words}")
print(f"Total entropy: {entropy}")

# Create an empty directed graph
G = nx.DiGraph()

# Split the text into words and build nodes for unique words
text = " ".join(data['cleaned_text'])  # Combine all resume text
words = text.split()
unique_words = set(words)

for word in unique_words:
    G.add_node(word)

# Define a co-occurrence threshold
co_occurrence_threshold = 5

# Create edges between words that co-occur within a certain window
for i in range(len(words)):
    for j in range(i + 1, min(i + co_occurrence_threshold, len(words))):
        if words[i] != words[j]:
            G.add_edge(words[i], words[j])


# Calculate node degree centrality
degree_centrality = nx.degree_centrality(G)

# Get the top 10 most central words
top_words = sorted(degree_centrality, key=degree_centrality.get, reverse=True)[:10]
print("Top 10 central words:", top_words)

# layout for graph visualization
layout = nx.spring_layout(G)

# Draw the nodes and edges
nx.draw(G, layout, with_labels=True, node_size=10, font_size=6, font_color='b')

# Display the graph
plt.title("Network Graph")
plt.axis('off')
plt.show()
