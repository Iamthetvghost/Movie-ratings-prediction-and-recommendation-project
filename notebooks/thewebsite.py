# -*- coding: utf-8 -*-
"""
Created on Sun Oct 18 21:40:12 2026

@author: Elena
"""

from pathlib import Path

import numpy as np
import pandas as pd

data_dir = Path("data/ml-32m")

movies = pd.read_csv(data_dir / "movies.csv")
ratings = pd.read_csv(data_dir / "ratings.csv")

print("Movies:", movies.shape)
print("Ratings:", ratings.shape)
print("Users:", ratings["userId"].nunique())
print("Unique rated movies:", ratings["movieId"].nunique())



from surprise import Dataset, Reader, SVD
reader = Reader(rating_scale=(0.5, 5.0))
full_trainset = Dataset.load_from_df(
    ratings[["userId", "movieId", "rating"]],
    reader
).build_full_trainset()
print("Number of ratings:", full_trainset.n_ratings)
print("Number of users:", full_trainset.n_users)
print("Number of movies:", full_trainset.n_items)
svd_model = SVD(
    n_factors=20,
    n_epochs=20,
    lr_all=0.005,
    reg_all=0.02,
    random_state=42
)
print("Training SVD...")
svd_model.fit(full_trainset)
print("Training completed.")


# Extract movie factors in the same order as the internal SVD item indices
internal_movie_ids = list(full_trainset.all_items())

movie_ids_original = np.array(
    [
        int(full_trainset.to_raw_iid(i))
        for i in internal_movie_ids
    ],
    dtype=np.int64
)

movie_factors = np.array(
    [svd_model.qi[i] for i in internal_movie_ids],
    dtype=np.float32
)

# Normalise each movie vector
norms = np.linalg.norm(movie_factors, axis=1, keepdims=True)

movie_factors_norm = movie_factors / np.maximum(norms, 1e-12)

# Map original MovieLens movie IDs to vector indices
movie_id_to_idx = {
    int(movie_id): idx
    for idx, movie_id in enumerate(movie_ids_original)
}

print("Movie vectors:", movie_factors_norm.shape)
print("Movies with trained factors:", len(movie_id_to_idx))


def find_movie_id(title):
    """Find a movie by exact title or an unambiguous partial title."""
    title = title.strip()

    if not title:
        raise ValueError("Please enter a movie title.")

    titles = movies["title"].fillna("")

    # Prefer an exact, case-insensitive match
    exact = movies[
        titles.str.casefold() == title.casefold()
    ]

    if not exact.empty:
        match = exact.iloc[0]
    else:
        matches = movies[
            titles.str.contains(title, case=False, regex=False, na=False)
        ]

        if matches.empty:
            raise ValueError(f"No movie found matching: {title}")

        if len(matches) > 1:
            suggestions = "\n".join(
                f"- {name}" for name in matches["title"].head(10)
            )
            raise ValueError(
                f"Multiple movies match '{title}'. Please be more specific:\n"
                f"{suggestions}"
            )

        match = matches.iloc[0]

    movie_id = int(match["movieId"])

    if movie_id not in movie_id_to_idx:
        raise ValueError(
            f"'{match['title']}' has no trained SVD vector."
        )

    return movie_id, match["title"]


def recommend_svd(movie_ids, top_n=10):
    """Recommend movies using average cosine similarity in SVD space."""
    if top_n < 1:
        raise ValueError("top_n must be at least 1.")

    input_ids = list(dict.fromkeys(int(i) for i in movie_ids))

    if not input_ids:
        raise ValueError("Please provide at least one input movie.")

    missing = [
        movie_id for movie_id in input_ids
        if movie_id not in movie_id_to_idx
    ]

    if missing:
        raise ValueError(f"No trained SVD vectors for movie IDs: {missing}")

    input_indices = np.array(
        [movie_id_to_idx[movie_id] for movie_id in input_ids]
    )

    # Each row is a candidate movie; each column is an input movie
    similarities = (
        movie_factors_norm
        @ movie_factors_norm[input_indices].T
    )

    scores = similarities.mean(axis=1)

    # Exclude the input movies
    scores[input_indices] = -np.inf

    top_indices = np.argsort(scores)[::-1][:top_n]

    results = pd.DataFrame({
        "movieId": movie_ids_original[top_indices],
        "svd_similarity": scores[top_indices]
    })

    results = results.merge(
        movies[["movieId", "title", "genres"]],
        on="movieId",
        how="left"
    )

    results["year"] = results["title"].str.extract(r"\((\d{4})\)\s*$")

    return results[["title", "year", "genres", "svd_similarity"]]


website_dir = Path("website")
website_data_dir = website_dir / "data"

website_data_dir.mkdir(parents=True, exist_ok=True)

# Save movie IDs and their corresponding normalised latent vectors
np.savez_compressed(
    website_data_dir / "movie_vectors.npz",
    movie_ids=movie_ids_original,
    vectors=movie_factors_norm
)

# Export metadata only for movies with trained SVD vectors
website_movies = movies[
    movies["movieId"].isin(movie_ids_original)
][["movieId", "title", "genres"]].copy()

website_movies["movieId"] = website_movies["movieId"].astype("int64")

website_movies.to_csv(
    website_data_dir / "movies.csv",
    index=False
)

print("Website data exported:")
print(" -", website_data_dir / "movie_vectors.npz")
print(" -", website_data_dir / "movies.csv")
print("Movies exported:", len(website_movies))


app_code = r'''
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

st.set_page_config(
    page_title="Film Affinity",
    page_icon="▪",
    layout="centered"
)

# Minimal dark interface
st.markdown(
    """
    <style>
    .stApp {
        background-color: #080808;
        color: #eeeeee;
    }
    .block-container {
        max-width: 760px;
        padding-top: 4rem;
        padding-bottom: 4rem;
    }
    h1, h2, h3, p, label {
        color: #eeeeee !important;
    }
    div[data-baseweb="input"] {
        background-color: #111111;
    }
    input {
        color: #eeeeee !important;
    }
    div.stButton > button {
        background-color: #eeeeee;
        color: #080808;
        border: 1px solid #eeeeee;
        border-radius: 2px;
        font-weight: 600;
        width: 100%;
    }
    div.stButton > button:hover {
        background-color: #cccccc;
        color: #080808;
        border-color: #cccccc;
    }
    hr {
        border-color: #333333;
    }
    .film-label {
        color: #888888;
        font-family: monospace;
        font-size: 0.8rem;
        letter-spacing: 0.08em;
    }
    .result-title {
        color: #eeeeee;
        font-size: 1rem;
        font-weight: 500;
    }
    .result-meta {
        color: #888888;
        font-size: 0.85rem;
    }
    </style>
    """,
    unsafe_allow_html=True
)


@st.cache_data
def load_data():
    metadata = pd.read_csv(DATA_DIR / "movies.csv")
    stored = np.load(DATA_DIR / "movie_vectors.npz")

    movie_ids = stored["movie_ids"].astype(np.int64)
    vectors = stored["vectors"].astype(np.float32)

    id_to_idx = {
        int(movie_id): idx
        for idx, movie_id in enumerate(movie_ids)
    }

    return metadata, movie_ids, vectors, id_to_idx


movies, movie_ids, vectors, id_to_idx = load_data()


def find_movie(query):
    query = query.strip()

    if not query:
        raise ValueError("Please enter a movie title.")

    titles = movies["title"].fillna("")

    exact = movies[
        titles.str.casefold() == query.casefold()
    ]

    if not exact.empty:
        match = exact.iloc[0]
    else:
        matches = movies[
            titles.str.contains(
                query,
                case=False,
                regex=False,
                na=False
            )
        ]

        if matches.empty:
            raise ValueError(
                f"No movie found matching '{query}'."
            )

        if len(matches) > 1:
            suggestions = "\n".join(
                f"- {title}"
                for title in matches["title"].head(10)
            )
            raise ValueError(
                f"'{query}' matches multiple movies. "
                f"Please enter a more specific title:\n{suggestions}"
            )

        match = matches.iloc[0]

    movie_id = int(match["movieId"])

    if movie_id not in id_to_idx:
        raise ValueError(
            f"No trained SVD vector is available for '{match['title']}'."
        )

    return movie_id, match["title"]


def get_recommendations(input_ids, top_n=10):
    input_indices = np.array([
        id_to_idx[movie_id] for movie_id in input_ids
    ])

    # Cosine similarities: normalised vectors have unit length
    similarities = vectors @ vectors[input_indices].T
    scores = similarities.mean(axis=1)

    # Do not recommend the input films themselves
    scores[input_indices] = -np.inf

    top_indices = np.argsort(scores)[::-1][:top_n]

    results = pd.DataFrame({
        "movieId": movie_ids[top_indices],
        "similarity": scores[top_indices]
    })

    results = results.merge(
        movies,
        on="movieId",
        how="left"
    )

    return results


st.markdown(
    '<div class="film-label">CLOCKGHOST / FILM ENGINE / 001</div>',
    unsafe_allow_html=True
)

st.title("film / affinity")

st.write(
    "Enter three films. Discover the connections between them."
)

st.divider()

with st.form("film_input"):
    st.markdown(
        '<div class="film-label">01 / INPUT FILMS</div>',
        unsafe_allow_html=True
    )

    film_1 = st.text_input(
        "Film 01",
        placeholder="e.g. Mulholland Drive (2001)"
    )
    film_2 = st.text_input(
        "Film 02",
        placeholder="e.g. Cure (1997)"
    )
    film_3 = st.text_input(
        "Film 03",
        placeholder="e.g. The Master (2012)"
    )

    submitted = st.form_submit_button(
        "GENERATE RECOMMENDATIONS"
    )


if submitted:
    queries = [film_1, film_2, film_3]

    if any(not query.strip() for query in queries):
        st.error("Please enter all three movie titles.")
    else:
        try:
            resolved = [find_movie(query) for query in queries]
            input_ids = [item[0] for item in resolved]
            input_titles = [item[1] for item in resolved]

            if len(set(input_ids)) != 3:
                st.error("Please enter three different movies.")
            else:
                st.divider()
                st.markdown(
                    '<div class="film-label">02 / YOUR FILMS</div>',
                    unsafe_allow_html=True
                )

                for title in input_titles:
                    st.write(f"— {title}")

                results = get_recommendations(input_ids, top_n=10)

                st.divider()
                st.markdown(
                    '<div class="film-label">03 / RECOMMENDED FILMS</div>',
                    unsafe_allow_html=True
                )

                for rank, row in enumerate(
                    results.itertuples(index=False), start=1
                ):
                    st.markdown(f"**{rank:02d} / {row.title}**")

                    genres = (
                        row.genres
                        if pd.notna(row.genres)
                        else "Genres unavailable"
                    )

                    st.caption(
                        f"{genres}  ·  similarity {row.similarity:.3f}"
                    )

        except ValueError as error:
            st.error(str(error))

st.divider()
st.caption(
    "Recommendations are based on cosine similarity between "
    "SVD movie latent factors. Similarity is not a measure of "
    "objective film quality."
)
'''

app_path = website_dir / "app.py"
app_path.write_text(app_code.strip() + "\n", encoding="utf-8")

print("Website application generated:", app_path)


# Website runtime dependencies
requirements_text = """streamlit
numpy
pandas
"""

(website_dir / "requirements.txt").write_text(
    requirements_text,
    encoding="utf-8"
)

# Force a dark theme
streamlit_dir = website_dir / ".streamlit"
streamlit_dir.mkdir(parents=True, exist_ok=True)

config_text = """[theme]
base = "dark"
backgroundColor = "#080808"
secondaryBackgroundColor = "#151515"
textColor = "#EEEEEE"
primaryColor = "#EEEEEE"
font = "monospace"
"""

(streamlit_dir / "config.toml").write_text(
    config_text,
    encoding="utf-8"
)

print("Created:")
print(" - website/requirements.txt")
print(" - website/.streamlit/config.toml")


required_files = [
    website_dir / "app.py",
    website_dir / "requirements.txt",
    website_dir / ".streamlit" / "config.toml",
    website_data_dir / "movie_vectors.npz",
    website_data_dir / "movies.csv"
]

for file_path in required_files:
    print(
        f"{'OK' if file_path.exists() else 'MISSING'}: {file_path}"
    )

print("\nWebsite data size:")
for file_path in [
    website_data_dir / "movie_vectors.npz",
    website_data_dir / "movies.csv"
]:
    if file_path.exists():
        print(
            f"{file_path.name}: "
            f"{file_path.stat().st_size / (1024 ** 2):.2f} MB"
        )
        
        
        


