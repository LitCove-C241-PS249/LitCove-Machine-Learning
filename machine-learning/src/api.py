from flask import Flask, request, jsonify
import pandas as pd
import json
from fuzzywuzzy import process
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from scipy.sparse import csr_matrix
import logging

app = Flask(__name__)
logging.basicConfig(level=logging.DEBUG)

# Load the dataframe and mappings
df = pd.read_csv('https://storage.googleapis.com/litcove-buckets/dataset/dataset.csv')
book2idx = pd.read_pickle('https://storage.googleapis.com/litcove-buckets/model/book2idx.pkl')

# Load dense vectors from JSON
tfidf_vectors_path = '../model/tfidf_vectors.json'
with open(tfidf_vectors_path, 'r') as f:
    dense_vector = np.array(json.load(f))

# Convert dense vectors back to sparse matrix
vector = csr_matrix(dense_vector)

# Load TF-IDF model components from JSON
tfidf_model_path = '../model/tfidf_model.json'
with open(tfidf_model_path, 'r') as f:
    tfidf_data = json.load(f)

# Recreate the TF-IDF vectorizer with the loaded data
tfidf = TfidfVectorizer()
tfidf.vocabulary_ = tfidf_data["vocabulary_"]
tfidf.idf_ = np.array(tfidf_data["idf_"])
tfidf.stop_words_ = set(tfidf_data["stop_words_"])

# Cosine similarity based recommendation
def recommended_books_cosine(title):
    try:
        idx = book2idx[title]
    except KeyError:
        logging.debug(f"Title '{title}' not found in book2idx mapping.")
        matches = process.extract(title, df['bookTitle'].tolist(), limit=1)
        if matches and matches[0][1] >= 80:
            similar_name = matches[0][0]
            logging.debug(f"Fuzzy match found: '{similar_name}' for title '{title}'")
            # Directly use the similar name to get recommendations
            idx = book2idx.get(similar_name)
            if idx is not None:
                query = vector[idx]
                scores = cosine_similarity(query, vector)
                scores = scores.flatten()
                recommended_idx = (-scores).argsort()[1:6]
                return {"recommendations": df['bookTitle'].iloc[recommended_idx].tolist()}
            else:
                return {"error": "Book does not exist"}
        return {"error": "Book does not exist"}
    query = vector[idx]
    scores = cosine_similarity(query, vector)
    scores = scores.flatten()
    recommended_idx = (-scores).argsort()[1:6]
    return {"recommendations": df['bookTitle'].iloc[recommended_idx].tolist()}

# Jaccard similarity based recommendation for genre
def find_recommendation_jaccard_genre(genre):
    genre_set = set([genre])
    temp = df.copy()

    # Pastikan cleaned_bookGenres adalah set #letak masalah disini
    temp['cleaned_bookGenres'] = temp['cleaned_bookGenres'].apply(lambda x: set(x.split(',')) if isinstance(x, str) else x)

    temp['score'] = temp['cleaned_bookGenres'].apply(lambda x: len(x.intersection(genre_set)) / len(x.union(genre_set)) if isinstance(x, set) else 0)
    temp = temp.sort_values(by='score', ascending=False)
    top_5_rows = temp.iloc[:5, :]
    return {"recommendations": top_5_rows['bookTitle'].tolist()}

@app.route('/recommend/title', methods=['GET'])
def recommend_by_title():
    title = request.args.get('title')
    if not title:
        return jsonify({"error": "Title parameter is required"}), 400
    result = recommended_books_cosine(title)
    return jsonify(result)

@app.route('/recommend/genre', methods=['GET'])
def recommend_by_genre():
    genre = request.args.get('genre')
    if not genre:
        return jsonify({"error": "Genre parameter is required"}), 400
    result = find_recommendation_jaccard_genre(genre)
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)
