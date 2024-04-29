import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

# Load the dataset
movies = pd.read_csv('movies.csv')

# Preprocess the data
movies['genres'] = movies['genres'].fillna('')
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(movies['genres'])

# Compute similarity scores
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

# Function to recommend movies
def recommend_movies(movie_title, cosine_sim=cosine_sim):
    idx = movies.loc[movies['title'] == movie_title].index[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:11]
    movie_indices = [i[0] for i in sim_scores]
    return movies['title'].iloc[movie_indices]

# Example: Get recommendations for a movie
movie_title = 'The Dark Knight'
recommendations = recommend_movies(movie_title)
print(recommendations)
