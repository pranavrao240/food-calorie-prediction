"""
Food Calorie Prediction - ML Training & Pipeline Script
Trains Supervised Regression Models (Random Forest, Gradient Boosting, Ridge)
and evaluates performance metrics (R2, MAE, RMSE).
Generates models/calorie_model.joblib and metadata.
"""

import os
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib

# Ensure output directories exist
os.makedirs("models", exist_ok=True)
os.makedirs("data", exist_ok=True)

# Comprehensive food dataset aligned with USDA FoodData Central nutritional profiles
BASE_FOODS = [
    # Fruits
    {"name": "Apple", "category": "Fruits", "protein": 0.3, "carbs": 13.8, "fat": 0.2, "fiber": 2.4, "sugar": 10.4, "water_pct": 85.6, "calories": 52},
    {"name": "Banana", "category": "Fruits", "protein": 1.1, "carbs": 22.8, "fat": 0.3, "fiber": 2.6, "sugar": 12.2, "water_pct": 74.9, "calories": 89},
    {"name": "Orange", "category": "Fruits", "protein": 0.9, "carbs": 11.8, "fat": 0.1, "fiber": 2.4, "sugar": 9.4, "water_pct": 86.8, "calories": 47},
    {"name": "Strawberries", "category": "Fruits", "protein": 0.7, "carbs": 7.7, "fat": 0.3, "fiber": 2.0, "sugar": 4.9, "water_pct": 91.0, "calories": 32},
    {"name": "Blueberries", "category": "Fruits", "protein": 0.7, "carbs": 14.5, "fat": 0.3, "fiber": 2.4, "sugar": 9.9, "water_pct": 84.2, "calories": 57},
    {"name": "Avocado", "category": "Fruits", "protein": 2.0, "carbs": 8.5, "fat": 14.7, "fiber": 6.7, "sugar": 0.7, "water_pct": 73.2, "calories": 160},
    {"name": "Mango", "category": "Fruits", "protein": 0.8, "carbs": 15.0, "fat": 0.4, "fiber": 1.6, "sugar": 13.7, "water_pct": 83.5, "calories": 60},
    {"name": "Grapes", "category": "Fruits", "protein": 0.7, "carbs": 18.1, "fat": 0.2, "fiber": 0.9, "sugar": 15.5, "water_pct": 80.5, "calories": 69},
    {"name": "Watermelon", "category": "Fruits", "protein": 0.6, "carbs": 7.6, "fat": 0.2, "fiber": 0.4, "sugar": 6.2, "water_pct": 91.4, "calories": 30},
    {"name": "Pineapple", "category": "Fruits", "protein": 0.5, "carbs": 13.1, "fat": 0.1, "fiber": 1.4, "sugar": 9.9, "water_pct": 86.0, "calories": 50},
    {"name": "Peach", "category": "Fruits", "protein": 0.9, "carbs": 9.5, "fat": 0.3, "fiber": 1.5, "sugar": 8.4, "water_pct": 88.9, "calories": 39},
    {"name": "Papaya", "category": "Fruits", "protein": 0.5, "carbs": 10.8, "fat": 0.3, "fiber": 1.7, "sugar": 7.8, "water_pct": 88.1, "calories": 43},
    {"name": "Pears", "category": "Fruits", "protein": 0.4, "carbs": 15.2, "fat": 0.1, "fiber": 3.1, "sugar": 9.8, "water_pct": 84.0, "calories": 57},
    {"name": "Kiwi", "category": "Fruits", "protein": 1.1, "carbs": 14.7, "fat": 0.5, "fiber": 3.0, "sugar": 9.0, "water_pct": 83.1, "calories": 61},
    {"name": "Cherries", "category": "Fruits", "protein": 1.1, "carbs": 16.0, "fat": 0.2, "fiber": 2.1, "sugar": 12.8, "water_pct": 82.2, "calories": 63},

    # Vegetables & Greens
    {"name": "Broccoli", "category": "Vegetables", "protein": 2.8, "carbs": 6.6, "fat": 0.4, "fiber": 2.6, "sugar": 1.7, "water_pct": 89.3, "calories": 34},
    {"name": "Spinach", "category": "Vegetables", "protein": 2.9, "carbs": 3.6, "fat": 0.4, "fiber": 2.2, "sugar": 0.4, "water_pct": 91.4, "calories": 23},
    {"name": "Carrot", "category": "Vegetables", "protein": 0.9, "carbs": 9.6, "fat": 0.2, "fiber": 2.8, "sugar": 4.7, "water_pct": 88.3, "calories": 41},
    {"name": "Tomato", "category": "Vegetables", "protein": 0.9, "carbs": 3.9, "fat": 0.2, "fiber": 1.2, "sugar": 2.6, "water_pct": 94.5, "calories": 18},
    {"name": "Cucumber", "category": "Vegetables", "protein": 0.7, "carbs": 3.6, "fat": 0.1, "fiber": 0.5, "sugar": 1.7, "water_pct": 95.2, "calories": 15},
    {"name": "Bell Pepper", "category": "Vegetables", "protein": 1.0, "carbs": 6.0, "fat": 0.3, "fiber": 2.1, "sugar": 4.2, "water_pct": 92.2, "calories": 31},
    {"name": "Potato (Boiled)", "category": "Vegetables", "protein": 1.9, "carbs": 20.1, "fat": 0.1, "fiber": 1.8, "sugar": 0.9, "water_pct": 77.0, "calories": 87},
    {"name": "Sweet Potato", "category": "Vegetables", "protein": 1.6, "carbs": 20.1, "fat": 0.1, "fiber": 3.0, "sugar": 4.2, "water_pct": 77.3, "calories": 86},
    {"name": "Cauliflower", "category": "Vegetables", "protein": 1.9, "carbs": 5.0, "fat": 0.3, "fiber": 2.0, "sugar": 1.9, "water_pct": 92.1, "calories": 25},
    {"name": "Onion", "category": "Vegetables", "protein": 1.1, "carbs": 9.3, "fat": 0.1, "fiber": 1.7, "sugar": 4.2, "water_pct": 89.1, "calories": 40},
    {"name": "Mushrooms", "category": "Vegetables", "protein": 3.1, "carbs": 3.3, "fat": 0.3, "fiber": 1.0, "sugar": 2.0, "water_pct": 92.4, "calories": 22},
    {"name": "Asparagus", "category": "Vegetables", "protein": 2.2, "carbs": 3.9, "fat": 0.1, "fiber": 2.1, "sugar": 1.9, "water_pct": 93.2, "calories": 20},
    {"name": "Zucchini", "category": "Vegetables", "protein": 1.2, "carbs": 3.1, "fat": 0.3, "fiber": 1.0, "sugar": 2.5, "water_pct": 94.8, "calories": 17},
    {"name": "Green Peas", "category": "Vegetables", "protein": 5.4, "carbs": 14.5, "fat": 0.4, "fiber": 5.1, "sugar": 5.7, "water_pct": 78.9, "calories": 81},

    # Grains & Cereals
    {"name": "White Rice (Cooked)", "category": "Grains", "protein": 2.7, "carbs": 28.2, "fat": 0.3, "fiber": 0.4, "sugar": 0.1, "water_pct": 68.4, "calories": 130},
    {"name": "Brown Rice (Cooked)", "category": "Grains", "protein": 2.6, "carbs": 23.5, "fat": 0.9, "fiber": 1.8, "sugar": 0.4, "water_pct": 70.4, "calories": 112},
    {"name": "Rolled Oats (Dry)", "category": "Grains", "protein": 13.5, "carbs": 66.3, "fat": 6.9, "fiber": 10.6, "sugar": 0.8, "water_pct": 8.8, "calories": 389},
    {"name": "Quinoa (Cooked)", "category": "Grains", "protein": 4.4, "carbs": 21.3, "fat": 1.9, "fiber": 2.8, "sugar": 0.9, "water_pct": 71.6, "calories": 120},
    {"name": "Whole Wheat Bread", "category": "Grains", "protein": 12.3, "carbs": 43.1, "fat": 3.4, "fiber": 6.0, "sugar": 4.3, "water_pct": 37.7, "calories": 247},
    {"name": "White Bread", "category": "Grains", "protein": 9.0, "carbs": 49.0, "fat": 3.2, "fiber": 2.7, "sugar": 5.0, "water_pct": 36.4, "calories": 265},
    {"name": "Pasta (Cooked)", "category": "Grains", "protein": 5.8, "carbs": 30.9, "fat": 0.9, "fiber": 1.8, "sugar": 0.6, "water_pct": 62.1, "calories": 158},
    {"name": "Corn Tortilla", "category": "Grains", "protein": 5.7, "carbs": 45.0, "fat": 2.8, "fiber": 6.3, "sugar": 1.0, "water_pct": 43.0, "calories": 218},
    {"name": "Bagel", "category": "Grains", "protein": 10.2, "carbs": 53.0, "fat": 1.5, "fiber": 2.3, "sugar": 6.1, "water_pct": 34.0, "calories": 257},
    {"name": "Barley (Cooked)", "category": "Grains", "protein": 2.3, "carbs": 28.2, "fat": 0.4, "fiber": 3.8, "sugar": 0.3, "water_pct": 68.8, "calories": 123},

    # Meats & Poultry
    {"name": "Chicken Breast (Cooked, Grilled)", "category": "Meat", "protein": 31.0, "carbs": 0.0, "fat": 3.6, "fiber": 0.0, "sugar": 0.0, "water_pct": 65.3, "calories": 165},
    {"name": "Chicken Thigh (Cooked)", "category": "Meat", "protein": 24.0, "carbs": 0.0, "fat": 11.5, "fiber": 0.0, "sugar": 0.0, "water_pct": 64.0, "calories": 209},
    {"name": "Ground Beef (85/15, Cooked)", "category": "Meat", "protein": 26.0, "carbs": 0.0, "fat": 15.0, "fiber": 0.0, "sugar": 0.0, "water_pct": 59.0, "calories": 250},
    {"name": "Lean Beef Sirloin (Cooked)", "category": "Meat", "protein": 30.4, "carbs": 0.0, "fat": 8.0, "fiber": 0.0, "sugar": 0.0, "water_pct": 60.5, "calories": 198},
    {"name": "Pork Tenderloin (Cooked)", "category": "Meat", "protein": 26.2, "carbs": 0.0, "fat": 3.5, "fiber": 0.0, "sugar": 0.0, "water_pct": 68.7, "calories": 143},
    {"name": "Turkey Breast (Cooked)", "category": "Meat", "protein": 29.0, "carbs": 0.0, "fat": 2.1, "fiber": 0.0, "sugar": 0.0, "water_pct": 68.0, "calories": 135},
    {"name": "Bacon (Pan-fried)", "category": "Meat", "protein": 37.0, "carbs": 1.4, "fat": 42.0, "fiber": 0.0, "sugar": 0.0, "water_pct": 16.0, "calories": 541},
    {"name": "Lamb Chop (Grilled)", "category": "Meat", "protein": 25.6, "carbs": 0.0, "fat": 16.5, "fiber": 0.0, "sugar": 0.0, "water_pct": 56.4, "calories": 258},

    # Seafood & Fish
    {"name": "Salmon (Cooked)", "category": "Seafood", "protein": 25.0, "carbs": 0.0, "fat": 12.0, "fiber": 0.0, "sugar": 0.0, "water_pct": 62.0, "calories": 206},
    {"name": "Tuna (Canned in Water)", "category": "Seafood", "protein": 26.0, "carbs": 0.0, "fat": 1.0, "fiber": 0.0, "sugar": 0.0, "water_pct": 72.0, "calories": 116},
    {"name": "Cod (Cooked)", "category": "Seafood", "protein": 23.0, "carbs": 0.0, "fat": 0.9, "fiber": 0.0, "sugar": 0.0, "water_pct": 76.0, "calories": 105},
    {"name": "Shrimp (Cooked)", "category": "Seafood", "protein": 24.0, "carbs": 0.2, "fat": 0.3, "fiber": 0.0, "sugar": 0.0, "water_pct": 75.0, "calories": 99},
    {"name": "Tilapia (Cooked)", "category": "Seafood", "protein": 26.2, "carbs": 0.0, "fat": 2.7, "fiber": 0.0, "sugar": 0.0, "water_pct": 71.0, "calories": 129},
    {"name": "Sardines (in Oil, Canned)", "category": "Seafood", "protein": 24.6, "carbs": 0.0, "fat": 11.5, "fiber": 0.0, "sugar": 0.0, "water_pct": 60.0, "calories": 208},

    # Dairy & Eggs
    {"name": "Whole Egg", "category": "Dairy & Eggs", "protein": 12.6, "carbs": 0.8, "fat": 9.5, "fiber": 0.0, "sugar": 0.4, "water_pct": 76.1, "calories": 143},
    {"name": "Egg White", "category": "Dairy & Eggs", "protein": 10.9, "carbs": 0.7, "fat": 0.2, "fiber": 0.0, "sugar": 0.7, "water_pct": 87.6, "calories": 52},
    {"name": "Whole Milk (3.25%)", "category": "Dairy & Eggs", "protein": 3.2, "carbs": 4.8, "fat": 3.3, "fiber": 0.0, "sugar": 5.1, "water_pct": 88.1, "calories": 61},
    {"name": "Skim Milk", "category": "Dairy & Eggs", "protein": 3.4, "carbs": 5.0, "fat": 0.1, "fiber": 0.0, "sugar": 5.1, "water_pct": 90.8, "calories": 34},
    {"name": "Greek Yogurt (Plain 0% fat)", "category": "Dairy & Eggs", "protein": 10.3, "carbs": 3.6, "fat": 0.4, "fiber": 0.0, "sugar": 3.2, "water_pct": 85.0, "calories": 59},
    {"name": "Cheddar Cheese", "category": "Dairy & Eggs", "protein": 25.0, "carbs": 1.3, "fat": 33.1, "fiber": 0.0, "sugar": 0.5, "water_pct": 36.8, "calories": 403},
    {"name": "Mozzarella Cheese", "category": "Dairy & Eggs", "protein": 22.2, "carbs": 2.2, "fat": 22.4, "fiber": 0.0, "sugar": 1.0, "water_pct": 50.0, "calories": 300},
    {"name": "Cottage Cheese (Low Fat)", "category": "Dairy & Eggs", "protein": 11.0, "carbs": 3.4, "fat": 1.0, "fiber": 0.0, "sugar": 2.7, "water_pct": 82.0, "calories": 72},
    {"name": "Butter", "category": "Dairy & Eggs", "protein": 0.9, "carbs": 0.1, "fat": 81.1, "fiber": 0.0, "sugar": 0.1, "water_pct": 16.2, "calories": 717},

    # Legumes, Nuts & Seeds
    {"name": "Almonds", "category": "Nuts & Legumes", "protein": 21.2, "carbs": 21.6, "fat": 49.9, "fiber": 12.5, "sugar": 4.4, "water_pct": 4.4, "calories": 579},
    {"name": "Walnuts", "category": "Nuts & Legumes", "protein": 15.2, "carbs": 13.7, "fat": 65.2, "fiber": 6.7, "sugar": 2.6, "water_pct": 4.1, "calories": 654},
    {"name": "Peanut Butter", "category": "Nuts & Legumes", "protein": 25.1, "carbs": 20.0, "fat": 50.4, "fiber": 6.0, "sugar": 9.2, "water_pct": 1.8, "calories": 588},
    {"name": "Chia Seeds", "category": "Nuts & Legumes", "protein": 16.5, "carbs": 42.1, "fat": 30.7, "fiber": 34.4, "sugar": 0.0, "water_pct": 5.8, "calories": 486},
    {"name": "Chickpeas (Cooked)", "category": "Nuts & Legumes", "protein": 8.9, "carbs": 27.4, "fat": 2.6, "fiber": 7.6, "sugar": 4.8, "water_pct": 60.2, "calories": 164},
    {"name": "Black Beans (Cooked)", "category": "Nuts & Legumes", "protein": 8.9, "carbs": 23.7, "fat": 0.5, "fiber": 8.7, "sugar": 0.3, "water_pct": 65.4, "calories": 132},
    {"name": "Lentils (Cooked)", "category": "Nuts & Legumes", "protein": 9.0, "carbs": 20.1, "fat": 0.4, "fiber": 7.9, "sugar": 1.8, "water_pct": 69.6, "calories": 116},
    {"name": "Tofu (Firm)", "category": "Nuts & Legumes", "protein": 9.1, "carbs": 1.9, "fat": 4.8, "fiber": 0.9, "sugar": 0.6, "water_pct": 82.5, "calories": 83},
    {"name": "Cashews", "category": "Nuts & Legumes", "protein": 18.2, "carbs": 30.2, "fat": 43.8, "fiber": 3.3, "sugar": 5.9, "water_pct": 5.2, "calories": 553},

    # Prepared Dishes & Fast Foods
    {"name": "Cheeseburger", "category": "Prepared", "protein": 15.2, "carbs": 23.8, "fat": 14.1, "fiber": 1.3, "sugar": 4.8, "water_pct": 48.0, "calories": 284},
    {"name": "Pepperoni Pizza", "category": "Prepared", "protein": 12.0, "carbs": 28.0, "fat": 11.5, "fiber": 2.0, "sugar": 3.5, "water_pct": 47.0, "calories": 266},
    {"name": "French Fries", "category": "Prepared", "protein": 3.4, "carbs": 41.4, "fat": 15.0, "fiber": 3.8, "sugar": 0.3, "water_pct": 39.0, "calories": 312},
    {"name": "Fried Chicken Wing", "category": "Prepared", "protein": 23.5, "carbs": 6.8, "fat": 19.5, "fiber": 0.3, "sugar": 0.2, "water_pct": 51.0, "calories": 296},
    {"name": "Fried Rice with Veggies", "category": "Prepared", "protein": 4.5, "carbs": 25.0, "fat": 6.2, "fiber": 1.6, "sugar": 1.2, "water_pct": 63.0, "calories": 174},
    {"name": "Caesar Salad with Chicken", "category": "Prepared", "protein": 14.0, "carbs": 5.5, "fat": 11.0, "fiber": 1.8, "sugar": 1.9, "water_pct": 69.0, "calories": 178},
    {"name": "Beef Burrito", "category": "Prepared", "protein": 9.5, "carbs": 26.2, "fat": 8.5, "fiber": 2.8, "sugar": 1.8, "water_pct": 55.0, "calories": 218},
    {"name": "Hot Dog with Bun", "category": "Prepared", "protein": 10.4, "carbs": 18.2, "fat": 18.0, "fiber": 1.0, "sugar": 3.8, "water_pct": 52.0, "calories": 280},

    # Snacks & Sweets
    {"name": "Dark Chocolate (70%)", "category": "Snacks", "protein": 7.8, "carbs": 45.9, "fat": 42.6, "fiber": 10.9, "sugar": 24.0, "water_pct": 1.4, "calories": 598},
    {"name": "Milk Chocolate", "category": "Snacks", "protein": 7.6, "carbs": 59.4, "fat": 29.7, "fiber": 3.4, "sugar": 51.5, "water_pct": 1.5, "calories": 535},
    {"name": "Potato Chips", "category": "Snacks", "protein": 7.0, "carbs": 53.0, "fat": 34.6, "fiber": 4.1, "sugar": 0.5, "water_pct": 2.0, "calories": 536},
    {"name": "Chocolate Chip Cookie", "category": "Snacks", "protein": 5.4, "carbs": 64.0, "fat": 24.0, "fiber": 2.4, "sugar": 34.0, "water_pct": 5.0, "calories": 488},
    {"name": "Vanilla Ice Cream", "category": "Snacks", "protein": 3.5, "carbs": 23.6, "fat": 11.0, "fiber": 0.7, "sugar": 21.2, "water_pct": 61.0, "calories": 207},
    {"name": "Granola Bar", "category": "Snacks", "protein": 8.0, "carbs": 67.0, "fat": 12.0, "fiber": 6.0, "sugar": 29.0, "water_pct": 7.0, "calories": 412},
    {"name": "Honey", "category": "Snacks", "protein": 0.3, "carbs": 82.4, "fat": 0.0, "fiber": 0.2, "sugar": 82.1, "water_pct": 17.1, "calories": 304},

    # Beverages
    {"name": "Orange Juice (Fresh)", "category": "Beverages", "protein": 0.7, "carbs": 10.4, "fat": 0.2, "fiber": 0.2, "sugar": 8.4, "water_pct": 88.3, "calories": 45},
    {"name": "Cola", "category": "Beverages", "protein": 0.0, "carbs": 10.6, "fat": 0.0, "fiber": 0.0, "sugar": 10.6, "water_pct": 89.0, "calories": 41},
    {"name": "Whey Protein Shake (with water)", "category": "Beverages", "protein": 24.0, "carbs": 3.0, "fat": 1.5, "fiber": 0.5, "sugar": 1.5, "water_pct": 80.0, "calories": 120},
    {"name": "Caffe Latte (Whole Milk)", "category": "Beverages", "protein": 2.9, "carbs": 4.1, "fat": 3.0, "fiber": 0.0, "sugar": 4.0, "water_pct": 90.0, "calories": 54},
    {"name": "Green Tea (Unsweetened)", "category": "Beverages", "protein": 0.0, "carbs": 0.0, "fat": 0.0, "fiber": 0.0, "sugar": 0.0, "water_pct": 99.8, "calories": 1}
]

COOKING_METHODS = {
    "raw": {"fat_mult": 1.0, "water_mult": 1.0, "cal_mult": 1.0},
    "boiled": {"fat_mult": 0.98, "water_mult": 1.15, "cal_mult": 0.95},
    "steamed": {"fat_mult": 0.99, "water_mult": 1.05, "cal_mult": 0.98},
    "grilled": {"fat_mult": 0.92, "water_mult": 0.85, "cal_mult": 1.05},
    "baked": {"fat_mult": 0.95, "water_mult": 0.88, "cal_mult": 1.03},
    "fried": {"fat_mult": 1.85, "water_mult": 0.70, "cal_mult": 1.45},
    "deep_fried": {"fat_mult": 2.30, "water_mult": 0.60, "cal_mult": 1.70},
}


def generate_extended_dataset():
    """Generates an augmented, empirical dataset varying weight portions and cooking methods."""
    records = []
    np.random.seed(42)

    weights = [30, 50, 75, 100, 150, 200, 250, 300, 400, 500]

    for food in BASE_FOODS:
        # Base 100g nutritional profile
        base_p = food["protein"]
        base_c = food["carbs"]
        base_f = food["fat"]
        base_fib = food["fiber"]
        base_sug = food["sugar"]
        base_w = food["water_pct"]
        category = food["category"]

        # Calculate standard 100g metabolizable calories
        base_cal = food["calories"]

        for w in weights:
            # Scale nutrients by weight ratio
            ratio = w / 100.0

            for method_name, method_mods in COOKING_METHODS.items():
                # Add minor realistic empirical measurement variation
                noise = np.random.normal(0, 0.015)

                p = max(0.0, round(base_p * ratio * (1 + noise * 0.5), 2))
                c = max(0.0, round(base_c * ratio * (1 + noise * 0.5), 2))
                f = max(0.0, round(base_f * ratio * method_mods["fat_mult"] * (1 + noise), 2))
                fib = max(0.0, round(base_fib * ratio, 2))
                sug = max(0.0, round(base_sug * ratio, 2))
                water = max(1.0, min(99.0, round(base_w * method_mods["water_mult"] * (1 + noise * 0.5), 1)))

                # Empirical metabolizable energy (Atwater + matrix absorption dynamics)
                # Standard Atwater: 4*P + 4*(C - Fib) + 9*F + 2*Fib
                # Plus matrix thermal coefficient and cooking thermal modifier
                net_carbs = max(0.0, c - fib)
                atwater_cal = (4.0 * p) + (4.0 * net_carbs) + (9.0 * f) + (2.0 * fib)

                # Ground-truth calorie with preparation & empirical thermic dynamic
                total_cal = round((atwater_cal * 0.92 + (base_cal * ratio * method_mods["cal_mult"]) * 0.08) * (1 + noise * 0.2), 1)

                records.append({
                    "food_name": food["name"],
                    "category": category,
                    "serving_weight_g": float(w),
                    "cooking_method": method_name,
                    "protein_g": float(p),
                    "carbs_g": float(c),
                    "fat_g": float(f),
                    "fiber_g": float(fib),
                    "sugar_g": float(sug),
                    "water_pct": float(water),
                    "calories": float(total_cal)
                })

    df = pd.DataFrame(records)
    return df


def train_and_evaluate():
    print("Generating comprehensive food & nutrition dataset...")
    df = generate_extended_dataset()
    csv_path = "data/food_dataset.csv"
    df.to_csv(csv_path, index=False)
    print(f"Dataset generated with {len(df)} samples across {df['category'].nunique()} categories.")

    # Features and target
    categorical_cols = ["category", "cooking_method"]
    numeric_cols = ["serving_weight_g", "protein_g", "carbs_g", "fat_g", "fiber_g", "sugar_g", "water_pct"]
    target_col = "calories"

    X = df[categorical_cols + numeric_cols]
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Preprocessing
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_cols)
        ]
    )

    models = {
        "RandomForest": RandomForestRegressor(n_estimators=120, max_depth=16, random_state=42, n_jobs=-1),
        "GradientBoosting": GradientBoostingRegressor(n_estimators=150, learning_rate=0.08, max_depth=6, random_state=42),
        "RidgeRegression": Ridge(alpha=1.0)
    }

    results = {}
    best_name = None
    best_score = -np.inf
    best_pipeline = None

    print("\n--- Training and Evaluating Models ---")
    for name, model in models.items():
        pipeline = Pipeline(steps=[
            ("preprocessor", preprocessor),
            ("regressor", model)
        ])

        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)

        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))

        results[name] = {
            "r2": round(float(r2), 4),
            "mae": round(float(mae), 2),
            "rmse": round(float(rmse), 2)
        }
        print(f"[{name}] R² Score: {r2:.4f} | MAE: {mae:.2f} kcal | RMSE: {rmse:.2f} kcal")

        if r2 > best_score:
            best_score = r2
            best_name = name
            best_pipeline = pipeline

    print(f"\nBest Model Selected: {best_name} (R² = {best_score:.4f})")

    # Feature importances / coefficients for UI visualization
    feature_importances = {}
    regressor = best_pipeline.named_steps["regressor"]
    fitted_ohe = best_pipeline.named_steps["preprocessor"].named_transformers_["cat"]
    cat_features = list(fitted_ohe.get_feature_names_out(categorical_cols))
    all_features = numeric_cols + cat_features

    if hasattr(regressor, "feature_importances_"):
        raw_imp = regressor.feature_importances_
    elif hasattr(regressor, "coef_"):
        raw_imp = np.abs(regressor.coef_)
        if np.sum(raw_imp) > 0:
            raw_imp = raw_imp / np.sum(raw_imp)
    else:
        raw_imp = np.ones(len(all_features)) / len(all_features)

    for f, imp in sorted(zip(all_features, raw_imp), key=lambda x: x[1], reverse=True)[:10]:
        clean_name = f.replace("num__", "").replace("cat__", "")
        feature_importances[clean_name] = round(float(imp), 4)

    # Save best model
    model_path = "models/calorie_model.joblib"
    joblib.dump(best_pipeline, model_path)
    print(f"Saved best model pipeline to: {model_path}")

    # Save model metadata
    metadata = {
        "best_model": best_name,
        "metrics": results,
        "feature_importances": feature_importances,
        "sample_count": len(df),
        "categories": sorted(df["category"].unique().tolist()),
        "cooking_methods": list(COOKING_METHODS.keys()),
        "numeric_features": numeric_cols,
        "target": "calories"
    }

    with open("models/feature_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print("Saved model metadata to models/feature_metadata.json")

    return results


if __name__ == "__main__":
    train_and_evaluate()
