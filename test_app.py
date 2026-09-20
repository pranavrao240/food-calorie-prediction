"""
Automated Test Suite for Food Calorie Prediction System
Validates Database operations, ML inference, NLP parsing, Clustering, and Flask REST APIs.
"""

import unittest
import json
import os
import joblib
import pandas as pd

from database import init_db, search_foods, get_all_foods, log_meal, get_meal_history
from nlp_parser import extract_quantity_and_unit, match_food_item, parse_meal_description
from clustering import perform_food_clustering
from app import app


class TestFoodCalorieSystem(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Initialize DB and ensure test client
        init_db()
        cls.client = app.test_client()
        cls.ml_model = joblib.load("models/calorie_model.joblib")

    def test_01_database_seed_and_search(self):
        """Verify database contains seeded USDA food items and search works correctly."""
        all_foods = get_all_foods()
        self.assertGreaterEqual(len(all_foods), 50, "Expected at least 50 seeded foods in DB")

        # Search query
        apple_results = search_foods("Apple")
        self.assertGreater(len(apple_results), 0)
        self.assertEqual(apple_results[0]["name"], "Apple")

        chicken_results = search_foods("Chicken")
        self.assertTrue(any("Chicken" in f["name"] for f in chicken_results))

    def test_02_ml_model_prediction(self):
        """Verify ML regression pipeline predicts consistent, non-negative caloric values."""
        sample_input = pd.DataFrame([{
            "category": "Meat",
            "cooking_method": "grilled",
            "serving_weight_g": 150.0,
            "protein_g": 46.5,
            "carbs_g": 0.0,
            "fat_g": 5.4,
            "fiber_g": 0.0,
            "sugar_g": 0.0,
            "water_pct": 65.0
        }])

        pred = self.ml_model.predict(sample_input)[0]
        self.assertGreater(pred, 150.0)
        self.assertLess(pred, 350.0)

    def test_03_nlp_quantity_and_unit_extraction(self):
        """Verify regex extractor parses quantities and units accurately."""
        qty, unit, rem = extract_quantity_and_unit("150g grilled chicken")
        self.assertEqual(qty, 150.0)
        self.assertEqual(unit, "g")
        self.assertEqual(rem, "grilled chicken")

        qty2, unit2, rem2 = extract_quantity_and_unit("2 cups of milk")
        self.assertEqual(qty2, 2.0)
        self.assertEqual(unit2, "cups")

        qty3, unit3, rem3 = extract_quantity_and_unit("2 boiled eggs")
        self.assertEqual(qty3, 2.0)
        self.assertEqual(unit3, "piece")

    def test_04_nlp_meal_parsing_pipeline(self):
        """Verify full meal sentence parsing, aggregation, and macro scaling."""
        meal_text = "2 boiled eggs, 100g oatmeal and 1 banana"
        result = parse_meal_description(meal_text, ml_pipeline=self.ml_model)

        self.assertEqual(result["item_count"], 3)
        self.assertGreater(result["total_calories"], 400.0)
        self.assertGreater(result["total_protein"], 15.0)
        self.assertGreater(result["total_carbs"], 60.0)

        # Check matched food names
        matched_names = [item["matched_food"] for item in result["items"]]
        self.assertIn("Whole Egg", matched_names)
        self.assertIn("Rolled Oats (Dry)", matched_names)
        self.assertIn("Banana", matched_names)

    def test_05_unsupervised_clustering(self):
        """Verify K-Means produces 5 clusters and 2D PCA projections."""
        cluster_res = perform_food_clustering(k=5)
        self.assertEqual(len(cluster_res["cluster_summaries"]), 5)
        self.assertGreaterEqual(len(cluster_res["points"]), 50)
        self.assertEqual(len(cluster_res["pca_variance_ratio"]), 2)

    def test_06_api_home_endpoint(self):
        """Verify home dashboard HTML loads."""
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"NutriCal AI", response.data)

    def test_07_api_predict_endpoint(self):
        """Verify POST /api/predict returns valid JSON predictions."""
        payload = {
            "food_name": "Salmon Fillet",
            "category": "Seafood",
            "cooking_method": "baked",
            "serving_weight_g": 170.0,
            "protein_g": 42.5,
            "carbs_g": 0.0,
            "fat_g": 20.4,
            "fiber_g": 0.0,
            "sugar_g": 0.0,
            "water_pct": 62.0
        }
        response = self.client.post("/api/predict", data=json.dumps(payload), content_type="application/json")
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertTrue(data["success"])
        self.assertGreater(data["predicted_calories"], 250)
        self.assertIn("atwater_calories", data)
        self.assertIn("macro_percentages", data)

    def test_08_api_parse_meal_endpoint(self):
        """Verify POST /api/parse-meal returns structured items and totals."""
        payload = {"text": "1 cup brown rice and 200g grilled salmon"}
        response = self.client.post("/api/parse-meal", data=json.dumps(payload), content_type="application/json")
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertTrue(data["success"])
        self.assertEqual(data["item_count"], 2)
        self.assertGreater(data["total_calories"], 300)

    def test_09_api_log_and_history(self):
        """Verify meal logging and history retrieval in SQLite."""
        payload = {
            "meal_name": "Test Breakfast",
            "total_calories": 450.0,
            "total_protein": 24.0,
            "total_carbs": 50.0,
            "total_fat": 12.0,
            "items": [{"name": "Eggs", "cal": 150}, {"name": "Toast", "cal": 300}]
        }
        post_resp = self.client.post("/api/log-meal", data=json.dumps(payload), content_type="application/json")
        self.assertEqual(post_resp.status_code, 200)
        post_data = post_resp.get_json()
        self.assertTrue(post_data["success"])

        # Check history
        get_resp = self.client.get("/api/history")
        self.assertEqual(get_resp.status_code, 200)
        get_data = get_resp.get_json()
        self.assertTrue(get_data["success"])
        self.assertGreater(len(get_data["history"]), 0)
        self.assertEqual(get_data["history"][0]["meal_name"], "Test Breakfast")

    def test_10_api_model_info(self):
        """Verify GET /api/model-info returns ML metrics and cluster data."""
        response = self.client.get("/api/model-info")
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertTrue(data["success"])
        self.assertIn("metadata", data)
        self.assertIn("clustering", data)


if __name__ == "__main__":
    unittest.main()
