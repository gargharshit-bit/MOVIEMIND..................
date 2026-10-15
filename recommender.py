
import pandas as pd

movies = pd.read_csv("movies.csv")
tags = pd.read_csv("tags.csv")

print("Movies Dataset:")
print(movies.head())

print("\nTags Dataset:")
print(tags.head())

print("\nMovies Shape:", movies.shape)
print("Tags Shape:", tags.shape)


# Combine all tags belonging to each movie
movie_tags = tags.groupby("movieId")["tag"].apply(
    lambda x: " ".join(x.astype(str))
).reset_index()

# Add tags to the movies dataset
movies = movies.merge(movie_tags, on="movieId", how="left")

# Replace missing tags with empty text
movies["tag"] = movies["tag"].fillna("")

# Combine genres and tags into one column
movies["content"] = movies["genres"] + " " + movies["tag"]

# Display the prepared movie data
print("\nPrepared Movie Data:")
print(movies[["title", "genres", "tag", "content"]].head())

print("\nTotal movies:", len(movies))


from sklearn.feature_extraction.text import TfidfVectorizer

# Convert movie content into numerical features
tfidf = TfidfVectorizer(stop_words="english")

tfidf_matrix = tfidf.fit_transform(movies["content"])

print("\nTF-IDF Matrix Shape:", tfidf_matrix.shape)


from sklearn.metrics.pairwise import cosine_similarity

# Calculate similarity between movies
similarity = cosine_similarity(tfidf_matrix)

print("\nSimilarity Matrix Shape:", similarity.shape)

# Check similarity of Toy Story with itself
print("Toy Story self-similarity:", similarity[0][0])





def get_recommendations(movie_name):
    # Find the movie index
    movie_indices = movies[
        movies["title"].str.contains(
            movie_name, case=False, na=False
        )
    ].index

    if len(movie_indices) == 0:
        return "Movie not found!"

    movie_index = movie_indices[0]

    # Calculate similarity only for the selected movie
    movie_similarity = cosine_similarity(
        tfidf_matrix[movie_index],
        tfidf_matrix
    ).flatten()

    # Sort movies by similarity score
    similar_indices = movie_similarity.argsort()[::-1]

    # Remove the selected movie itself
    similar_indices = [
        i for i in similar_indices if i != movie_index
    ]

    # Return the top 5 similar movies
    return movies.iloc[similar_indices[:5]]["title"].tolist()


# Test the recommendation function
print("\nRecommended movies for Toy Story:")
print(get_recommendations("Toy Story"))
