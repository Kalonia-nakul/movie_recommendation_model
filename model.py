import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

# -------------------------------
# Load datasets
# -------------------------------
movies_df = pd.read_csv("movies.csv")
ratings_df = pd.read_csv("ratings.csv")

# -------------------------------
# Merge datasets
# -------------------------------
data = pd.merge(ratings_df, movies_df, on="movieId", how="left")

# -------------------------------
# Top rated movies (optional analysis)
# -------------------------------
high_rated = data.groupby('title')['rating'].sum().nlargest(20)
rating_count_20 = data.groupby('title')['rating'].count().nlargest(20)

print("Top 20 High Rated Movies:\n", high_rated)
print("\nTop 20 Most Rated Movies:\n", rating_count_20)

# -------------------------------
# Preprocess genres
# -------------------------------
movies_df['genres'] = movies_df['genres'].fillna('')
movies_df['genres'] = movies_df['genres'].str.replace('|', ' ', regex=False)

# -------------------------------
# TF-IDF Vectorization
# -------------------------------
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(movies_df['genres'])

# -------------------------------
# Compute similarity matrix
# -------------------------------
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

# -------------------------------
# Create index mapping
# -------------------------------
indices = pd.Series(movies_df.index, index=movies_df['title']).drop_duplicates()

# -------------------------------
# Recommendation Function
# -------------------------------
def recommend_movies(title, top_n=10):
    if title not in indices:
        return "❌ Movie not found in dataset"
    
    idx = indices[title]

    # Get similarity scores
    sim_scores = list(enumerate(cosine_sim[idx]))

    # Sort movies based on similarity
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get top N similar movies (excluding itself)
    sim_scores = sim_scores[1:top_n + 1]

    movie_indices = [i[0] for i in sim_scores]

    return movies_df['title'].iloc[movie_indices]

# -------------------------------
# Example Usage
# -------------------------------
movie_name = "Toy Story (1995)"
recommendations = recommend_movies(movie_name)

print(f"\nRecommendations for '{movie_name}':\n")
print(recommendations)