
import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="MovieMind", page_icon="🎬")

@st.cache_data
def load_data():
    movies = pd.read_csv("movies.csv")
    tags = pd.read_csv("tags.csv")

    movie_tags = tags.groupby("movieId")["tag"].apply(
        lambda x: " ".join(x.astype(str))
    ).reset_index()

    movies = movies.merge(movie_tags, on="movieId", how="left")
    movies["tag"] = movies["tag"].fillna("")
    movies["content"] = movies["genres"].fillna("") + " " + movies["tag"]

    tfidf = TfidfVectorizer(stop_words="english")
    tfidf_matrix = tfidf.fit_transform(movies["content"])

    return movies, tfidf_matrix


movies, tfidf_matrix = load_data()


def get_recommendations(movie_name):
    movie_indices = movies[
        movies["title"].str.contains(
            movie_name, case=False, na=False, regex=False
        )
    ].index

    if len(movie_indices) == 0:
        return None, []

    movie_index = movie_indices[0]

    movie_similarity = cosine_similarity(
        tfidf_matrix[movie_index], tfidf_matrix
    ).flatten()

    similar_indices = movie_similarity.argsort()[::-1]
    similar_indices = [
        i for i in similar_indices if i != movie_index
    ]

    recommendations = movies.iloc[similar_indices[:5]]["title"].tolist()

    return movies.iloc[movie_index]["title"], recommendations


st.title("🎬 MovieMind")
st.write("Discover movies similar to your favourite movie!")

movie_name = st.text_input(
    "Enter a movie name",
    placeholder="e.g. Toy Story"
)

if st.button("Recommend Movies"):
    if not movie_name.strip():
        st.warning("Please enter a movie name.")
    else:
        selected_movie, recommendations = get_recommendations(movie_name)

        if selected_movie is None:
            st.error("Movie not found! Try another movie name.")
        else:
            st.subheader(f"Movies similar to {selected_movie}")

            for i, movie in enumerate(recommendations, start=1):
                st.write(f"{i}. {movie}")
