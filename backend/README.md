# CineMatch backend

FastAPI backend for the CineMatch frontend. It follows the recommendation methods in `CINEMATCH.ipynb`: genre TF-IDF cosine similarity, mean-centered user-user collaborative filtering with similarity shrinkage, and the notebook's mood/genre reranking weights.

## Setup

From this directory, activate the existing environment and install dependencies if needed:

```powershell
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

The dataset is read from `..\ml-latest-small` by default. Set `CINEMATCH_DATA_DIR` to another directory containing `movies.csv` and `ratings.csv` when needed. The model is built once during application startup and held in memory.

## Run

```powershell
uvicorn main:app --reload --host 127.0.0.1 --port 8000
```

The API allows browser requests from `http://localhost:3000`.

## Endpoints

- `GET /health` reports service and model load state.
- `GET /movies?search=toy` searches titles and genre strings. Returns `{ "movies": [...] }` using the frontend `MovieReference` shape.
- `POST /recommend-by-mood` accepts `mood`, optional `genre` (null, empty, or `Any Genre` means mood only), and `top_k` (1–50, default 12). It returns mood-matched MovieLens movies ranked using mood alignment, rating quality, and rating count; a selected genre is enforced as a filter.
- `POST /recommend` accepts `mood`, `user_id`, optional `reference_movie_id` or `reference_movie_title`, optional `genre`, and `top_k` (1–50). The notebook hybrid recommender requires an existing MovieLens user. With neither reference supplied, it uses the notebook's `Toy Story (1995)` default. Unknown users, moods, and movies return HTTP errors.
- `GET /similar?movie_id=1` returns genre TF-IDF cosine neighbors.
- `GET /metrics` reports counts and sparsity calculated from the loaded dataset. `collaborative_rmse` is the notebook's recorded value (0.8586); the TF-IDF vocabulary size is reported as `embedding_dimensions` to match the current frontend field. `average_latency_ms` is measured from requests served by this process and is 0 before the first recommendation request. No training timestamp is supplied because the notebook does not record one.

Unavailable dataset fields (poster image, runtime, plot overview) are returned as null or omitted; the API does not invent them. Recommendation scores are computed from the notebook logic.

## Tests

Run the backend tests with:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```
