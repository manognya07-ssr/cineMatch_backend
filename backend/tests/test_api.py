import unittest
from fastapi import HTTPException

import main


class CineMatchApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        main.model = main.CineMatchModel()

    @classmethod
    def tearDownClass(cls):
        main.model = None

    def test_health_reports_loaded_model(self):
        self.assertEqual(main.health()["status"], "ok")

    def test_search_returns_frontend_movie_reference_shape(self):
        result = main.search_movies("Toy Story", limit=8)
        self.assertTrue(result["movies"])
        movie = result["movies"][0]
        self.assertTrue({"id", "title", "genres", "poster_url"}.issubset(movie))

    def test_recommendation_uses_real_user_and_reference_movie(self):
        result = main.recommend(main.RecommendRequest(mood="Happy", user_id=1,
                                                      reference_movie_title="Toy Story (1995)", top_k=3))
        self.assertLessEqual(len(result["recommendations"]), 3)
        self.assertEqual(result["query_summary"]["reference_movie"], "Toy Story (1995)")
        for movie in result["recommendations"]:
            self.assertTrue({"movie_id", "title", "genres", "match_score", "score_breakdown"}.issubset(movie))

    def test_recommend_by_mood_with_genre(self):
        for mood, genre in (("Happy", "Comedy"), ("Sad", "Drama")):
            result = main.recommend_by_mood(main.MoodRecommendRequest(mood=mood, genre=genre, top_k=12))
            self.assertEqual(result["mood"], mood)
            self.assertEqual(result["genre"], genre)
            self.assertGreater(result["total_recommendations"], 0)
            self.assertLessEqual(result["total_recommendations"], 12)
            for movie in result["recommendations"]:
                self.assertTrue({"movie_id", "title", "year", "genres", "rating", "match_score"}.issubset(movie))
                self.assertIn(genre, movie["genres"])

    def test_recommend_by_mood_without_genre_and_invalid_mood(self):
        result = main.recommend_by_mood(main.MoodRecommendRequest(mood="Happy", genre=None))
        self.assertIsNone(result["genre"])
        self.assertTrue(result["recommendations"])
        with self.assertRaises(HTTPException) as error:
            main.recommend_by_mood(main.MoodRecommendRequest(mood="Cheerful"))
        self.assertEqual(error.exception.status_code, 422)

    def test_recommend_by_mood_rejects_invalid_genre(self):
        with self.assertRaises(HTTPException) as error:
            main.recommend_by_mood(main.MoodRecommendRequest(mood="Happy", genre="NotAGenre"))
        self.assertEqual(error.exception.status_code, 422)
    def test_similar_excludes_source(self):
        result = main.similar(movie_id=1, limit=3)
        self.assertEqual(result["source_movie"]["id"], 1)
        self.assertTrue(all(m["movie_id"] != 1 for m in result["similar_movies"]))

    def test_metrics_are_based_on_loaded_data(self):
        result = main.metrics()
        self.assertEqual(result["n_movies"], int(main.model.movies.movieId.nunique()))
        self.assertEqual(result["n_users"], int(main.model.user_movie.index.nunique()))
        self.assertEqual(result["n_ratings"], len(main.model.ratings))
        self.assertGreater(result["embedding_dimensions"], 0)


if __name__ == "__main__":
    unittest.main()

