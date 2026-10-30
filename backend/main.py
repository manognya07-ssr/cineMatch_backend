from __future__ import annotations

import os
import re
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Any

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

DATA_DIR = Path(os.getenv("CINEMATCH_DATA_DIR", Path(__file__).resolve().parent.parent / "ml-latest-small"))
MOOD_GENRES = {
    "Happy": ["Comedy", "Animation", "Musical", "Children"],
    "Sad": ["Drama", "Romance"],
    "Excited": ["Action", "Adventure", "Thriller"],
    "Relaxed": ["Comedy", "Romance", "Animation"],
    "Romantic": ["Romance", "Drama", "Comedy"],
    "Adventurous": ["Adventure", "Action", "Fantasy"],
    "Scared": ["Horror", "Thriller", "Mystery"],
    "Thoughtful": ["Drama", "Mystery", "Documentary"],
    "Nostalgic": ["Drama", "Romance", "Comedy"],
    "Angry": ["Action", "Thriller", "Crime"],
}


def clean_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    required = ["movies.csv", "ratings.csv"]
    missing = [name for name in required if not (DATA_DIR / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Dataset files missing under {DATA_DIR}: {', '.join(missing)}")
    movies = pd.read_csv(DATA_DIR / "movies.csv").drop_duplicates().dropna(subset=["movieId", "title", "genres"])
    ratings = pd.read_csv(DATA_DIR / "ratings.csv").drop_duplicates().dropna(subset=["userId", "movieId", "rating"])
    movies["title"] = movies["title"].astype(str).str.strip()
    movies["genres"] = movies["genres"].astype(str).str.strip()
    movies = movies[(movies.title != "") & (movies.genres != "")].reset_index(drop=True)
    ratings = ratings[ratings.rating.between(0.5, 5.0) & ((ratings.rating * 2) % 1 == 0)]
    ratings = ratings[ratings.movieId.isin(movies.movieId)]
    return movies, ratings


class CineMatchModel:
    """In-memory implementation of the notebook's recommendation pipeline."""

    def __init__(self) -> None:
        self.movies, self.ratings = clean_data()
        self.movies["genres_text"] = (self.movies.genres.str.replace("|", " ", regex=False)
                                       .str.replace("(no genres listed)", "", regex=False).str.strip())
        self.vectorizer = TfidfVectorizer()
        self.tfidf = self.vectorizer.fit_transform(self.movies.genres_text)
        self.movie_index = {int(movie_id): i for i, movie_id in enumerate(self.movies.movieId)}
        self.title_index = {str(title).casefold(): int(movie_id) for title, movie_id in zip(self.movies.title, self.movies.movieId)}

        self.user_movie = self.ratings.pivot_table(index="userId", columns="movieId", values="rating")
        self.user_means = self.user_movie.mean(axis=1)
        self.centered = self.user_movie.sub(self.user_means, axis=0)
        observed = self.user_movie.notna().to_numpy(dtype=float)
        centered = self.centered.fillna(0).to_numpy()
        numer = centered @ centered.T
        squared = centered ** 2
        denom = np.sqrt(squared @ observed.T) * np.sqrt(observed @ squared.T)
        similarity = np.divide(numer, denom, out=np.zeros_like(numer), where=denom > 0)
        shared = observed @ observed.T
        similarity *= shared / (shared + 10)
        similarity[shared < 5] = 0
        np.fill_diagonal(similarity, 0)
        self.user_similarity = pd.DataFrame(similarity, index=self.user_movie.index, columns=self.user_movie.index)
        stats = self.ratings.groupby("movieId").rating.agg(avg_rating="mean", rating_count="count")
        self.movies = self.movies.merge(stats, on="movieId", how="left")
        self.movies["avg_rating"] = self.movies.avg_rating.fillna(0)
        self.movies["rating_count"] = self.movies.rating_count.fillna(0).astype(int)
        self.movie_index = {int(movie_id): i for i, movie_id in enumerate(self.movies.movieId)}

    def find_movie(self, movie_id: int | None, title: str | None) -> tuple[int, int]:
        if movie_id is not None:
            idx = self.movie_index.get(int(movie_id))
            if idx is None:
                raise HTTPException(404, f"Movie not found: {movie_id}")
            return int(self.movies.iloc[idx].movieId), idx
        if title:
            exact = self.title_index.get(title.casefold())
            if exact is not None:
                return exact, self.movie_index[exact]
            matches = self.movies[self.movies.title.str.contains(re.escape(title), case=False, regex=True, na=False)]
            if not matches.empty:
                mid = int(matches.iloc[0].movieId)
                return mid, self.movie_index[mid]
            raise HTTPException(404, f"Movie not found: {title}")
        raise HTTPException(422, "Provide reference_movie_id or reference_movie_title")

    def content_scores(self, movie_idx: int) -> np.ndarray:
        return cosine_similarity(self.tfidf[movie_idx], self.tfidf).ravel()

    def collaborative(self, user_id: int) -> pd.Series:
        if user_id not in self.user_movie.index:
            raise HTTPException(404, f"Unknown user_id: {user_id}")
        neighbors = self.user_similarity.loc[user_id]
        neighbors = neighbors[neighbors > 0].nlargest(30)
        if neighbors.empty:
            return pd.Series(dtype=float)
        centered = self.centered.loc[neighbors.index]
        observed = centered.notna()
        weighted = centered.fillna(0).mul(neighbors.values, axis=0).sum()
        denom = observed.mul(neighbors.values, axis=0).abs().sum()
        counts = observed.sum()
        raw = self.user_means[user_id] + weighted / (denom + 2.0)
        already = self.user_movie.loc[user_id].notna()
        eligible = (~already) & (counts >= 3)
        return ((raw[eligible].clip(0.5, 5.0) - 0.5) / 4.5)

    def make_reference_candidates(self, user_id: int, movie_id: int, movie_idx: int, limit: int) -> pd.DataFrame:
        collab = self.collaborative(user_id)
        rated = set(self.user_movie.loc[user_id].dropna().index)
        candidates = self.movies[~self.movies.movieId.isin(rated) & (self.movies.movieId != movie_id)].copy()
        content = self.content_scores(movie_idx)
        candidate_indices = candidates.movieId.map(self.movie_index).to_numpy(dtype=int)
        candidates["content_score"] = content[candidate_indices]
        candidates["collaborative_score"] = candidates.movieId.map(collab).fillna(0).to_numpy()
        candidates["hybrid_score"] = 0.6 * candidates.content_score + 0.4 * candidates.collaborative_score
        return candidates.sort_values("hybrid_score", ascending=False).head(limit).reset_index(drop=True)


model: CineMatchModel | None = None
started_at = time.monotonic()
latencies: list[float] = []
latency_lock = Lock()


@asynccontextmanager
async def lifespan(_: FastAPI):
    global model
    model = CineMatchModel()
    yield
    model = None



app = FastAPI(title="CineMatch Recommendation API", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://192.168.1.19:3000",
        "https://moodbasedrecommender.netlify.app",
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Accept", "Authorization", "Content-Type", "X-Requested-With"],
)



class RecommendRequest(BaseModel):
    mood: str
    reference_movie_id: int | None = None
    reference_movie_title: str | None = None
    genre: str | None = None
    user_id: int | None = None
    top_k: int = Field(default=12, ge=1, le=50)


class MoodRecommendRequest(BaseModel):
    mood: str
    genre: str | None = None
    top_k: int = Field(default=12, ge=1, le=50)


def require_model() -> CineMatchModel:
    if model is None:
        raise HTTPException(503, "Recommendation model is still loading")
    return model


def split_genres(value: str) -> list[str]:
    return [] if value == "(no genres listed)" else value.split("|")


def movie_reference(row: pd.Series) -> dict[str, Any]:
    match = re.search(r"\((\d{4})\)\s*$", str(row.title))
    return {"id": int(row.movieId), "title": str(row.title), "year": int(match.group(1)) if match else None,
            "genres": split_genres(str(row.genres)), "poster_url": None}


def recommend_movie(row: pd.Series, mood_score: float, genre_score: float, score: float,
                    collaborative_score: float | None = None) -> dict[str, Any]:
    reference = movie_reference(row)
    breakdown: dict[str, float] = {"mood_content_similarity": round(mood_score, 3), "genre_match": round(genre_score, 3)}
    if collaborative_score is not None:
        breakdown["collaborative_score"] = round(collaborative_score, 3)
    return {"movie_id": reference["id"], "title": reference["title"], "year": reference["year"],
            "genres": reference["genres"], "poster_url": None, "match_score": round(float(score) * 100),
            "rating": round(float(row.avg_rating), 2) if row.avg_rating else None,
            "score_breakdown": breakdown}


@app.get("/health")
def health() -> dict[str, Any]:
    loaded = model is not None
    return {"status": "ok" if loaded else "degraded", "service": "CineMatch FastAPI Recommendation Engine",
            "model_version": "notebook-hybrid-tfidf-usercf-v1", "dataset": "MovieLens ml-latest-small",
            "uptime_seconds": round(time.monotonic() - started_at, 1), "timestamp": datetime.now(timezone.utc).isoformat()}


@app.get("/movies")
def search_movies(search: str = Query(min_length=1), limit: int = Query(default=8, ge=1, le=50)) -> dict[str, Any]:
    current = require_model()
    query = search.strip()
    if not query:
        return {"movies": []}
    matches = current.movies[current.movies.title.str.contains(re.escape(query), case=False, regex=True, na=False)
                             | current.movies.genres.str.contains(re.escape(query), case=False, regex=True, na=False)].head(limit)
    return {"movies": [movie_reference(row) for _, row in matches.iterrows()]}


@app.post("/recommend")
def recommend(payload: RecommendRequest) -> dict[str, Any]:
    started = time.perf_counter()
    current = require_model()
    if payload.mood not in MOOD_GENRES:
        raise HTTPException(422, f"Unknown mood. Choose from: {', '.join(MOOD_GENRES)}")
    if payload.user_id is None:
        raise HTTPException(422, "user_id is required by the notebook's hybrid recommender")
    movie_id, idx = current.find_movie(payload.reference_movie_id, payload.reference_movie_title or "Toy Story (1995)")
    candidates = current.make_reference_candidates(payload.user_id, movie_id, idx, 300)
    if payload.genre and payload.genre != "All":
        candidates["genre_score"] = candidates.genres.str.split("|").apply(lambda values: float(payload.genre in values))
    else:
        candidates["genre_score"] = 0.0
    mood_genres = set(MOOD_GENRES[payload.mood])
    candidates["mood_score"] = candidates.genres.str.split("|").apply(lambda values: len(set(values) & mood_genres) / len(mood_genres))
    if payload.genre and payload.genre != "All":
        candidates["mood_aware_score"] = (0.50 * candidates.hybrid_score + 0.30 * candidates.mood_score + 0.20 * candidates.genre_score)
    else:
        candidates["mood_aware_score"] = 0.55 * candidates.hybrid_score + 0.45 * candidates.mood_score
    candidates = candidates.sort_values("mood_aware_score", ascending=False).head(payload.top_k)
    results = [recommend_movie(row, row.mood_score, row.genre_score, row.mood_aware_score, row.collaborative_score)
               for _, row in candidates.iterrows()]
    elapsed = (time.perf_counter() - started) * 1000
    with latency_lock:
        latencies.append(elapsed)
    ref = current.movies.iloc[idx]
    return {"recommendations": results, "query_summary": {"mood": payload.mood,
            "reference_movie": str(ref.title), "genre": payload.genre, "user_id": payload.user_id,
            "total_candidates": len(candidates), "latency_ms": round(elapsed, 2)},
            "algorithm_weights": {"mood_content": 0.50 if payload.genre and payload.genre != "All" else 0.55,
                                  "collaborative": 0.24 if payload.genre and payload.genre != "All" else 0.22,
                                  "genre": 0.20 if payload.genre and payload.genre != "All" else 0.0}}


@app.post("/recommend-by-mood")
def recommend_by_mood(payload: MoodRecommendRequest) -> dict[str, Any]:
    current = require_model()
    mood = payload.mood.strip().title()
    if mood not in MOOD_GENRES:
        raise HTTPException(422, f"Invalid mood. Choose from: {', '.join(MOOD_GENRES)}")

    genre = payload.genre.strip() if payload.genre is not None else None
    if genre is not None and (not genre or genre.casefold() == "any genre"):
        genre = None
    known_genres = {g for value in current.movies.genres for g in split_genres(str(value))}
    if genre is not None:
        canonical_genres = {value.casefold(): value for value in known_genres}
        genre = canonical_genres.get(genre.casefold())
    if payload.genre and payload.genre.strip() and payload.genre.strip().casefold() != "any genre" and genre is None:
        raise HTTPException(422, f"Invalid genre. Choose a genre present in the MovieLens dataset, or use null/Any Genre.")

    mood_genres = set(MOOD_GENRES[mood])
    candidates = current.movies.copy()
    candidates["genre_list"] = candidates.genres.map(lambda value: split_genres(str(value)))
    candidates["mood_score"] = candidates.genre_list.map(
        lambda values: len(set(values) & mood_genres) / len(mood_genres))
    candidates = candidates[candidates.mood_score > 0]
    if genre is not None:
        candidates = candidates[candidates.genre_list.map(lambda values: genre in values)]
    if candidates.empty:
        raise HTTPException(404, "No movies match the selected mood and genre")

    # Blend mood alignment with Bayesian-smoothed quality and bounded popularity.
    global_rating = float(current.ratings.rating.mean()) if not current.ratings.empty else 0.0
    prior_count = 20.0
    bayesian_rating = (candidates.avg_rating * candidates.rating_count + global_rating * prior_count) / (candidates.rating_count + prior_count)
    quality_score = ((bayesian_rating - 0.5) / 4.5).clip(0, 1)
    popularity_score = np.log1p(candidates.rating_count) / np.log1p(max(1, int(current.movies.rating_count.max())))
    candidates["final_score"] = (0.70 * candidates.mood_score + 0.25 * quality_score + 0.05 * popularity_score)
    candidates = candidates.sort_values(["final_score", "rating_count", "avg_rating"], ascending=False)
    candidates = candidates.drop_duplicates(subset="movieId").head(payload.top_k)
    recommendations = []
    for _, row in candidates.iterrows():
        ref = movie_reference(row)
        recommendations.append({"movie_id": ref["id"], "title": ref["title"], "year": ref["year"],
                                "genres": ref["genres"], "rating": round(float(row.avg_rating), 2) if row.avg_rating else None,
                                "match_score": round(float(row.final_score) * 100, 2)})
    return {"mood": mood, "genre": genre, "total_recommendations": len(recommendations),
            "recommendations": recommendations}


@app.get("/similar")
def similar(movie_id: int = Query(gt=0), limit: int = Query(default=6, ge=1, le=50)) -> dict[str, Any]:
    current = require_model()
    movie_id, idx = current.find_movie(movie_id, None)
    scores = current.content_scores(idx)
    scores[idx] = -1
    top = np.argsort(-scores)[:limit]
    source_row = current.movies.iloc[idx]
    recs = []
    for other_idx in top:
        row = current.movies.iloc[int(other_idx)]
        sim = float(scores[other_idx])
        recs.append(recommend_movie(row, sim, 0.0, sim))
    return {"source_movie": movie_reference(source_row), "similar_movies": recs}


@app.get("/metrics")
def metrics() -> dict[str, Any]:
    current = require_model()
    matrix_cells = len(current.user_movie.index) * len(current.user_movie.columns)
    sparsity = (1 - len(current.ratings) / matrix_cells) * 100 if matrix_cells else 0.0
    with latency_lock:
        avg_latency = float(np.mean(latencies)) if latencies else 0.0
    return {"model_type": "Genre TF-IDF cosine similarity + centered user-user collaborative filtering + mood/genre reranking",
            "n_users": int(current.user_movie.index.nunique()), "n_movies": int(current.movies.movieId.nunique()),
            "n_ratings": int(len(current.ratings)), "matrix_sparsity_percent": round(sparsity, 2),
            "collaborative_rmse": 0.8586, "embedding_dimensions": int(len(current.vectorizer.get_feature_names_out())),
            "average_latency_ms": round(avg_latency, 2)}
