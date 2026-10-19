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
st.title("cinema / affinity")
st.write(
    "Enter three films. Discover the connections between them."
)

st.markdown("""
<div style="font-size: 0.72rem; color: #888; line-height: 1.6; margin-top: -0.3rem; margin-bottom: 1.2rem;">
    <p>If a title begins with an article, move it to the end after a comma
    (e.g. <i>The Matrix</i> → <i>Matrix, The</i>).</p>
    <p>Start without the release year. If multiple matches appear, add a space
    followed by the year (e.g. <i>Crash (1996)</i>).</p>
</div>
""", unsafe_allow_html=True)

st.divider()

with st.form("film_input"):
    st.markdown(
        '<div class="film-label">01 / INPUT FILMS</div>',
        unsafe_allow_html=True
    )

    film_1 = st.text_input(
        "Film 01",
        placeholder="e.g. Mulholland Drive"
    )
    film_2 = st.text_input(
        "Film 02",
        placeholder="e.g. Meshes of the Afternoon"
    )
    film_3 = st.text_input(
        "Film 03",
        placeholder="e.g. Shining, The"
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
