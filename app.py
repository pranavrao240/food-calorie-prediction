"""
Food Calorie Prediction - Flask Web Application Backend
Provides RESTful APIs for ML calorie inference, NLP meal parsing,
unsupervised nutritional clustering, and SQLite meal logging.
"""

import os
os.environ["LOKY_MAX_CPU_COUNT"] = "4"

import json
import joblib
import pandas as pd
from flask import Flask, render_template, request, jsonify

from database import (
    init_db,
    search_foods,
    get_all_foods,
    get_food_by_name,
    log_meal,
    get_meal_history,
    log_prediction
)
from nlp_parser import parse_meal_description
from clustering import perform_food_clustering

app = Flask(__name__)

# Initialize database schema & seed data
init_db()

# Load trained ML pipeline & metadata
MODEL_PATH = os.path.join("models", "calorie_model.joblib")
META_PATH = os.path.join("models", "feature_metadata.json")

ml_pipeline = None
model_metadata = {}

if os.path.exists(MODEL_PATH):
    try:
        ml_pipeline = joblib.load(MODEL_PATH)
        print("ML Calorie Model successfully loaded.")
    except Exception as e:
        print(f"Error loading model: {e}")

if os.path.exists(META_PATH):
    try:
        with open(META_PATH, "r", encoding="utf-8") as f:
            model_metadata = json.load(f)
    except Exception as e:
        print(f"Error loading model metadata: {e}")


@app.route("/")
def home():
    """Renders main dashboard interface."""
    return render_template("index.html")


@app.route("/api/predict", methods=["POST"])
def predict_calories():
    """
    Predicts calories from food attributes and macronutrient values using ML.
    Compares against classical Atwater empirical factors.
    """
    data = request.get_json() or {}

    try:
        category = data.get("category", "General")
        cooking_method = data.get("cooking_method", "raw")
        weight_g = float(data.get("serving_weight_g", 100.0))
        protein_g = float(data.get("protein_g", 0.0))
        carbs_g = float(data.get("carbs_g", 0.0))
        fat_g = float(data.get("fat_g", 0.0))
        fiber_g = float(data.get("fiber_g", 0.0))
        sugar_g = float(data.get("sugar_g", 0.0))
        water_pct = float(data.get("water_pct", 70.0))
        food_name = data.get("food_name", "Custom Food").strip()

        # Classical Atwater Energy benchmark
        net_carbs = max(0.0, carbs_g - fiber_g)
        atwater_energy = (4.0 * protein_g) + (4.0 * net_carbs) + (9.0 * fat_g) + (2.0 * fiber_g)

        # ML Model Prediction
        if ml_pipeline is not None:
            input_df = pd.DataFrame([{
                "category": category,
                "cooking_method": cooking_method,
                "serving_weight_g": weight_g,
                "protein_g": protein_g,
                "carbs_g": carbs_g,
                "fat_g": fat_g,
                "fiber_g": fiber_g,
                "sugar_g": sugar_g,
                "water_pct": water_pct
            }])
            ml_pred = float(ml_pipeline.predict(input_df)[0])
            predicted_cal = max(1.0, round(ml_pred, 1))
        else:
            predicted_cal = round(atwater_energy, 1)

        # Energy distribution percentages
        protein_cals = protein_g * 4.0
        carbs_cals = net_carbs * 4.0
        fat_cals = fat_g * 9.0
        total_macro_cals = max(0.1, protein_cals + carbs_cals + fat_cals)

        macro_pct = {
            "protein": round((protein_cals / total_macro_cals) * 100, 1),
            "carbs": round((carbs_cals / total_macro_cals) * 100, 1),
            "fat": round((fat_cals / total_macro_cals) * 100, 1)
        }

        # Log prediction
        log_prediction(f"{food_name} ({weight_g}g)", predicted_cal, model_metadata.get("best_model", "RidgeRegression"))

        return jsonify({
            "success": True,
            "food_name": food_name,
            "predicted_calories": predicted_cal,
            "atwater_calories": round(atwater_energy, 1),
            "serving_weight_g": weight_g,
            "cooking_method": cooking_method,
            "macros": {
                "protein_g": round(protein_g, 1),
                "carbs_g": round(carbs_g, 1),
                "fat_g": round(fat_g, 1),
                "fiber_g": round(fiber_g, 1),
                "sugar_g": round(sugar_g, 1)
            },
            "macro_percentages": macro_pct,
            "model_used": model_metadata.get("best_model", "Trained ML Regressor")
        })

    except Exception as exc:
        return jsonify({"success": False, "error": str(exc)}), 400


@app.route("/api/parse-meal", methods=["POST"])
def parse_meal_api():
    """
    Natural Language Processing endpoint to parse multi-item meals or recipes,
    identifying food names, portions, preparation methods, and aggregate nutrition.
    """
    data = request.get_json() or {}
    text = data.get("text", "").strip()

    if not text:
        return jsonify({"success": False, "error": "Meal text is required."}), 400

    try:
        parsed_result = parse_meal_description(text, ml_pipeline=ml_pipeline)
        return jsonify({"success": True, **parsed_result})
    except Exception as exc:
        return jsonify({"success": False, "error": str(exc)}), 500


@app.route("/api/foods", methods=["GET"])
def get_foods():
    """Search or list food items from the database."""
    query = request.args.get("q", "").strip()
    if query:
        results = search_foods(query, limit=15)
    else:
        results = get_all_foods()
    return jsonify({"success": True, "foods": results})


@app.route("/api/log-meal", methods=["POST"])
def log_meal_api():
    """Save an assembled meal and its items to SQLite database."""
    data = request.get_json() or {}
    meal_name = data.get("meal_name", "Logged Meal").strip()
    total_calories = float(data.get("total_calories", 0.0))
    total_protein = float(data.get("total_protein", 0.0))
    total_carbs = float(data.get("total_carbs", 0.0))
    total_fat = float(data.get("total_fat", 0.0))
    items = data.get("items", [])

    try:
        log_id = log_meal(meal_name, total_calories, total_protein, total_carbs, total_fat, items)
        return jsonify({"success": True, "log_id": log_id, "message": "Meal successfully saved to diary."})
    except Exception as exc:
        return jsonify({"success": False, "error": str(exc)}), 500


@app.route("/api/history", methods=["GET"])
def history_api():
    """Retrieve logged meal history."""
    try:
        history = get_meal_history(limit=30)
        return jsonify({"success": True, "history": history})
    except Exception as exc:
        return jsonify({"success": False, "error": str(exc)}), 500


@app.route("/api/model-info", methods=["GET"])
def model_info_api():
    """Returns ML model benchmarks, feature importances, and unsupervised clusters."""
    try:
        clustering_data = perform_food_clustering(k=5)
        return jsonify({
            "success": True,
            "metadata": model_metadata,
            "clustering": clustering_data
        })
    except Exception as exc:
        return jsonify({"success": False, "error": str(exc)}), 500


if __name__ == "__main__":
    app.run(debug=True, port=5000)
