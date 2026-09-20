"""
Database management module for Food Calorie Prediction system.
Handles SQLite schema creation, seeding USDA-aligned food data,
and managing meal history & prediction logs.
"""

import sqlite3
import json
import os
from datetime import datetime, timedelta

DB_PATH = os.path.join("data", "food_calorie.db")


def get_connection():
    os.makedirs("data", exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_connection()
    cur = conn.cursor()

    # Foods table
    cur.execute("""
    CREATE TABLE IF NOT EXISTS foods (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT UNIQUE NOT NULL,
        category TEXT NOT NULL,
        serving_unit TEXT NOT NULL,
        standard_weight_g REAL NOT NULL,
        calories REAL NOT NULL,
        protein_g REAL NOT NULL,
        carbs_g REAL NOT NULL,
        fat_g REAL NOT NULL,
        fiber_g REAL NOT NULL,
        sugar_g REAL NOT NULL,
        water_pct REAL NOT NULL
    )
    """)

    # Meal logs table
    cur.execute("""
    CREATE TABLE IF NOT EXISTS meal_logs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        meal_name TEXT NOT NULL,
        logged_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        total_calories REAL NOT NULL,
        total_protein REAL NOT NULL,
        total_carbs REAL NOT NULL,
        total_fat REAL NOT NULL,
        items_json TEXT NOT NULL
    )
    """)

    # Predictions log table
    cur.execute("""
    CREATE TABLE IF NOT EXISTS predictions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        input_text TEXT,
        predicted_calories REAL NOT NULL,
        model_used TEXT NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """)

    conn.commit()

    # Seed foods if table is empty
    cur.execute("SELECT COUNT(*) as count FROM foods")
    count = cur.fetchone()["count"]

    if count == 0:
        seed_foods(cur)
        conn.commit()

    conn.close()


def seed_foods(cur):
    """Populates standard foods with common portion sizes and per-100g nutrient values."""
    from train_model import BASE_FOODS

    # Common portion descriptors and standard weights in grams
    portion_mapping = {
        "Apple": ("1 medium", 182),
        "Banana": ("1 medium", 118),
        "Orange": ("1 medium", 131),
        "Strawberries": ("1 cup whole", 152),
        "Blueberries": ("1 cup", 148),
        "Avocado": ("1 whole", 200),
        "Mango": ("1 medium", 200),
        "Grapes": ("1 cup", 150),
        "Watermelon": ("1 wedge", 286),
        "Pineapple": ("1 cup chunks", 165),
        "Peach": ("1 medium", 150),
        "Papaya": ("1 small", 157),
        "Pears": ("1 medium", 178),
        "Kiwi": ("1 medium", 69),
        "Cherries": ("1 cup", 138),
        "Broccoli": ("1 cup chopped", 91),
        "Spinach": ("2 cups raw", 60),
        "Carrot": ("1 medium", 61),
        "Tomato": ("1 medium", 123),
        "Cucumber": ("1/2 cup slices", 52),
        "Bell Pepper": ("1 medium", 119),
        "Potato (Boiled)": ("1 medium", 173),
        "Sweet Potato": ("1 medium", 114),
        "Cauliflower": ("1 cup pieces", 107),
        "Onion": ("1 medium", 110),
        "Mushrooms": ("1 cup sliced", 70),
        "Asparagus": ("6 spears", 90),
        "Zucchini": ("1 medium", 196),
        "Green Peas": ("1 cup", 160),
        "White Rice (Cooked)": ("1 cup", 158),
        "Brown Rice (Cooked)": ("1 cup", 195),
        "Rolled Oats (Dry)": ("1/2 cup dry", 40),
        "Quinoa (Cooked)": ("1 cup", 185),
        "Whole Wheat Bread": ("1 slice", 43),
        "White Bread": ("1 slice", 38),
        "Pasta (Cooked)": ("1 cup cooked", 140),
        "Corn Tortilla": ("1 piece", 26),
        "Bagel": ("1 regular", 105),
        "Barley (Cooked)": ("1 cup", 157),
        "Chicken Breast (Cooked, Grilled)": ("1 breast fillet", 172),
        "Chicken Thigh (Cooked)": ("1 thigh bone-out", 110),
        "Ground Beef (85/15, Cooked)": ("1 patty (4 oz)", 113),
        "Lean Beef Sirloin (Cooked)": ("1 steak (6 oz)", 170),
        "Pork Tenderloin (Cooked)": ("1 cutlet (3 oz)", 85),
        "Turkey Breast (Cooked)": ("3 slices (3 oz)", 85),
        "Bacon (Pan-fried)": ("3 cooked strips", 24),
        "Lamb Chop (Grilled)": ("1 chop (4 oz)", 113),
        "Salmon (Cooked)": ("1 fillet (6 oz)", 170),
        "Tuna (Canned in Water)": ("1 can drained", 165),
        "Cod (Cooked)": ("1 fillet (4 oz)", 113),
        "Shrimp (Cooked)": ("10 large shrimp", 145),
        "Tilapia (Cooked)": ("1 fillet (4 oz)", 113),
        "Sardines (in Oil, Canned)": ("1 can drained", 92),
        "Whole Egg": ("1 large egg", 50),
        "Egg White": ("1 large egg white", 33),
        "Whole Milk (3.25%)": ("1 cup (240ml)", 244),
        "Skim Milk": ("1 cup (240ml)", 245),
        "Greek Yogurt (Plain 0% fat)": ("1 container (7 oz)", 200),
        "Cheddar Cheese": ("1 slice", 28),
        "Mozzarella Cheese": ("1/4 cup shredded", 28),
        "Cottage Cheese (Low Fat)": ("1/2 cup", 113),
        "Butter": ("1 tablespoon", 14),
        "Almonds": ("1 handful (1 oz)", 28),
        "Walnuts": ("1 handful (1 oz)", 28),
        "Peanut Butter": ("2 tablespoons", 32),
        "Chia Seeds": ("2 tablespoons", 24),
        "Chickpeas (Cooked)": ("1 cup", 164),
        "Black Beans (Cooked)": ("1 cup", 172),
        "Lentils (Cooked)": ("1 cup", 198),
        "Tofu (Firm)": ("1/2 block", 126),
        "Cashews": ("1 handful (1 oz)", 28),
        "Cheeseburger": ("1 single burger", 150),
        "Pepperoni Pizza": ("1 slice (1/8 of 14-inch)", 107),
        "French Fries": ("1 medium order", 117),
        "Fried Chicken Wing": ("1 wing with skin", 45),
        "Fried Rice with Veggies": ("1 bowl", 200),
        "Caesar Salad with Chicken": ("1 serving bowl", 250),
        "Beef Burrito": ("1 regular burrito", 280),
        "Hot Dog with Bun": ("1 hot dog", 100),
        "Dark Chocolate (70%)": ("3 squares", 30),
        "Milk Chocolate": ("1 regular bar", 43),
        "Potato Chips": ("1 single-serve bag", 28),
        "Chocolate Chip Cookie": ("1 medium cookie", 30),
        "Vanilla Ice Cream": ("1/2 cup scoop", 66),
        "Granola Bar": ("1 bar", 40),
        "Honey": ("1 tablespoon", 21),
        "Orange Juice (Fresh)": ("1 glass (250ml)", 248),
        "Cola": ("1 can (355ml)", 370),
        "Whey Protein Shake (with water)": ("1 scoop + water", 300),
        "Caffe Latte (Whole Milk)": ("1 tall cup (350ml)", 350),
        "Green Tea (Unsweetened)": ("1 mug (240ml)", 240)
    }

    for food in BASE_FOODS:
        name = food["name"]
        unit, weight = portion_mapping.get(name, ("100g serving", 100.0))
        cur.execute("""
            INSERT OR IGNORE INTO foods (
                name, category, serving_unit, standard_weight_g,
                calories, protein_g, carbs_g, fat_g, fiber_g, sugar_g, water_pct
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            name,
            food["category"],
            unit,
            weight,
            food["calories"],
            food["protein"],
            food["carbs"],
            food["fat"],
            food["fiber"],
            food["sugar"],
            food["water_pct"]
        ))


def search_foods(query, limit=10):
    conn = get_connection()
    cur = conn.cursor()
    search_term = f"%{query.strip()}%"
    cur.execute("""
        SELECT * FROM foods 
        WHERE name LIKE ? OR category LIKE ?
        ORDER BY 
            CASE WHEN name LIKE ? THEN 0 ELSE 1 END,
            name ASC
        LIMIT ?
    """, (search_term, search_term, f"{query.strip()}%", limit))
    rows = cur.fetchall()
    conn.close()
    return [dict(row) for row in rows]


def get_food_by_name(name):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT * FROM foods WHERE LOWER(name) = LOWER(?)", (name.strip(),))
    row = cur.fetchone()
    conn.close()
    return dict(row) if row else None


def get_all_foods():
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT * FROM foods ORDER BY category ASC, name ASC")
    rows = cur.fetchall()
    conn.close()
    return [dict(row) for row in rows]


def log_meal(meal_name, total_calories, total_protein, total_carbs, total_fat, items):
    conn = get_connection()
    cur = conn.cursor()
    items_json = json.dumps(items)
    cur.execute("""
        INSERT INTO meal_logs (meal_name, total_calories, total_protein, total_carbs, total_fat, items_json)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (meal_name, round(total_calories, 1), round(total_protein, 1), round(total_carbs, 1), round(total_fat, 1), items_json))
    log_id = cur.lastrowid
    conn.commit()
    conn.close()
    return log_id


def get_meal_history(limit=25):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("""
        SELECT id, meal_name, logged_at, total_calories, total_protein, total_carbs, total_fat, items_json
        FROM meal_logs
        ORDER BY logged_at DESC
        LIMIT ?
    """, (limit,))
    rows = cur.fetchall()
    conn.close()
    results = []
    for r in rows:
        d = dict(r)
        try:
            d["items"] = json.loads(d["items_json"])
        except Exception:
            d["items"] = []
        results.append(d)
    return results


def log_prediction(input_text, predicted_calories, model_used="RidgeRegression"):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO predictions (input_text, predicted_calories, model_used)
        VALUES (?, ?, ?)
    """, (input_text, round(predicted_calories, 1), model_used))
    pred_id = cur.lastrowid
    conn.commit()
    conn.close()
    return pred_id
