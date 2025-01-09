#%% 
import pandas as pd
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from collections import Counter
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer
from textblob import TextBlob
import nltk
import gutenbergpy.textget
#%%

# Download NLTK data
nltk.download('punkt')
nltk.download('stopwords')
#%%

# Sample text
animal_house = {
    "text": """Over? Did you say ‘over?!’ Nothing is over until we decide it is! Was it over when the Germans bombed Pearl Harbor? Hell no! And it ain’t over now..."""
}

animal_house_df = pd.DataFrame([animal_house])

# Tokenize words
animal_house_df["tokens"] = animal_house_df["text"].apply(word_tokenize)

#%%
# View tokenized words
print(animal_house_df["tokens"])

# Tokenize sentences
animal_house_df["sentences"] = animal_house_df["text"].apply(sent_tokenize)

# Generate n-grams
def generate_ngrams(text, n=3):
    tokens = word_tokenize(text)
    ngrams = zip(*[tokens[i:] for i in range(n)])
    return [" ".join(gram) for gram in ngrams]

animal_house_df["ngrams"] = animal_house_df["text"].apply(lambda x: generate_ngrams(x, n=3))

# Count word frequencies
def count_words(tokens):
    return Counter(tokens)

animal_house_df["word_counts"] = animal_house_df["tokens"].apply(count_words)

# Remove stopwords
stop_words = set(stopwords.words('english'))
animal_house_df["filtered_tokens"] = animal_house_df["tokens"].apply(lambda x: [word for word in x if word.lower() not in stop_words])

# Recount after removing stopwords
animal_house_df["filtered_word_counts"] = animal_house_df["filtered_tokens"].apply(count_words)


#%%
# Quick plot of word frequencies
def plot_word_frequencies(word_counts):
    df = pd.DataFrame(word_counts.most_common(10), columns=["Word", "Frequency"])
    df.plot.bar(x="Word", y="Frequency", legend=False)
    plt.show()

#%%
# Plot word frequencies
plot_word_frequencies(animal_house_df["filtered_word_counts"][0])

# Word cloud
def generate_wordcloud(word_counts):
    wordcloud = WordCloud(width=800, height=400).generate_from_frequencies(word_counts)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.show()

generate_wordcloud(animal_house_df["filtered_word_counts"][0])

# Sentiment analysis using TextBlob
def sentiment_analysis(text):
    blob = TextBlob(text)
    return blob.sentiment

animal_house_df["sentiment"] = animal_house_df["text"].apply(sentiment_analysis)
print(animal_house_df["sentiment"])

#%%
import nltk
nltk.download('gutenberg')

file_ids = nltk.corpus.gutenberg.fileids()

keyword = "dracula"
matching_titles = [file_id for file_id in file_ids if keyword.lower() in file_id.lower()]

print(f"\nTitles containing the keyword '{keyword}':")
for title in matching_titles:
    print(title)

# Load raw text from the Gutenberg corpus
# raw = nltk.corpus.gutenberg.raw('dracula.txt')

# Print the first 50 lines of the raw text
# print('\n'.join(raw.split('\n')[:50]))
#%%

# TF-IDF for multiple documents
texts = [
    "Kennedy inaugural address text here...",
    "Trump inaugural address text here...",
    "Biden inaugural address text here..."
]
presidents = ["Kennedy", "Trump", "Biden"]

tfidf_vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf_vectorizer.fit_transform(texts)

# Create DataFrame for TF-IDF
tfidf_df = pd.DataFrame(tfidf_matrix.T.toarray(), index=tfidf_vectorizer.get_feature_names_out(), columns=presidents)
print(tfidf_df)

# Histogram of sentiment
def plot_sentiment_histogram(sentiments, title):
    plt.hist([s.polarity for s in sentiments], bins=10)
    plt.title(title)
    plt.show()

sentiments = animal_house_df["sentiment"].apply(lambda x: x.polarity)
plot_sentiment_histogram(sentiments, "Sentiment Analysis")

