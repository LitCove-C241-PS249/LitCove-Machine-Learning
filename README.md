# LitCove
# Book Recommendation System

Litcove is a book recommendation system that suggests books based on titles and genres. Using cosine similarity and Jaccard similarity, Litcove provides personalized book recommendations. The system is built using Python and Flask, and the dataset used is the Goodreads Best Books Ever dataset.


## Table of Contents
- [Introduction](#introduction)
- [Data Preparation and EDA](#data-preparation-and-eda)
- [Model Building](#model-building)
- [Flask API](#flask-api)
- [Installation](#installation)
- [Usage](#usage)
- [Acknowledgments](#acknowledgments)

## Introduction

The book recommendation system is designed to recommend books based on user input. It leverages cosine similarity for title-based recommendations and Jaccard similarity for genre-based recommendations. The dataset used is `Goodreads_BestBooksEver_1-10000` containing data of about 10,000 books.

## Data Preparation and EDA

1. **Loading Data**: The dataset is loaded and initial exploration is conducted.
2. **Cleaning Data**: Handle missing values, duplicates, and extract genres from the `bookGenres` column.
3. **Language Detection**: Use the `langid` library to detect the language of book titles.

## Model Building

1. **TF-IDF Vectorization**: Convert combined genres and language data into numerical data.
2. **Cosine Similarity**: Measure similarity between book titles.
3. **Jaccard Similarity**: Measure similarity between genres.

### Functions

- `recommended_books_cosine(title)`: Recommend books based on the cosine similarity of titles.
- `find_recommendation_jaccard_genre(genre)`: Recommend books based on the Jaccard similarity of genres.
- `find_recommendation_jaccard_title(name)`: Recommend books based on the Jaccard similarity of titles.

## Flask API

The Flask API provides endpoints to get book recommendations:

- **Title Recommendation**: `/recommend/title?title=<book_title>`
- **Genre Recommendation**: `/recommend/genre?genre=<genre_name>`

### Endpoints

- `GET /recommend/title`: Get book recommendations based on the title.
- `GET /recommend/genre`: Get book recommendations based on the genre.

## Installation

1. **Clone the repository**:
   ```bash
   git clone <repository_url>
   cd <repository_directory>
