import pandas as pd
import numpy as np
import re
import math
from collections import Counter

def load_movies(filepath):
    df = pd.read_csv(filepath)
    df['genres'] = df['genres'].fillna('')
    df['description'] = df['description'].fillna('')
    df['content'] = df['genres'].str.replace('|', ' ', regex=False) + ' ' + df['description']
    return df

def tokenize(text):
    text = text.lower()
    words = re.findall(r'\b\w+\b', text)
    return [word for word in words if len(word) > 2]

class SimpleTfidfVectorizer:
    def __init__(self):
        self.vocab = {}
        self.idf = {}

    def fit_transform(self, documents):
        tokenized_docs = [tokenize(doc) for doc in documents]
        df_counts = Counter()
        
        for doc in tokenized_docs:
            unique_words = set(doc)
            for word in unique_words:
                df_counts[word] += 1

        N = len(documents)
        self.vocab = {word: idx for idx, word in enumerate(df_counts)}
        self.idf = {word: math.log(N / (1 + df_counts[word])) for word in df_counts}
        
        tfidf_matrix = np.zeros((N, len(self.vocab)))
        for i, doc in enumerate(tokenized_docs):
            word_counts = Counter(doc)
            for word, count in word_counts.items():
                if word in self.vocab:
                    tf = count / len(doc)
                    tfidf_matrix[i][self.vocab[word]] = tf * self.idf[word]
                    
        return tfidf_matrix

def cosine_similarity(vec_a, vec_b):
    dot = np.dot(vec_a, vec_b)
    norm_a = np.linalg.norm(vec_a)
    norm_b = np.linalg.norm(vec_b)
    return dot / (norm_a * norm_b + 1e-10)

def recommend_movies(df, tfidf_matrix, watched_titles, top_n=5):
    indices = []
    for title in watched_titles:
        match = df[df['title'] == title]
        if not match.empty:
            indices.append(match.index[0])
        else:
            print(f"[!] Movie '{title}' not found.")
            
    if not indices:
        print("[!] No valid movies found.")
        return pd.DataFrame()

    avg_vector = np.mean(tfidf_matrix[indices], axis=0)
    scores = []
    
    for i in range(len(df)):
        if i in indices:
            continue
        sim = cosine_similarity(avg_vector, tfidf_matrix[i])
        scores.append((i, sim))
        
    top_indices = sorted(scores, key=lambda x: x[1], reverse=True)[:top_n]
    selected = [i for i, _ in top_indices]
    return df.iloc[selected][['title', 'genres', 'description']]

# --- Main Execution ---
df = load_movies('movies.csv')
vectorizer = SimpleTfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(df['content'])

print("Enter a movie name and it's release year, for example : Heat (1995), Jumanji (1995), etc...")
watched_movies = []
for i in range(2):
    wm = input("Enter the movie name : ")
    watched_movies.append(wm)

# watched_movies = ["Grumpier Old Men (1995)", "Waiting to Exhale (1995)"]
recommended_df = recommend_movies(df, tfidf_matrix, watched_movies)

if recommended_df.empty:
    print("No recommendations found.")
else:
    print("\nRecommended Movies:\n")
    for _, row in recommended_df.iterrows():
        print(f"Title: {row['title']}")
        print(f"Genres: {row['genres']}")
        print(f"Description: {row['description']}\n")
