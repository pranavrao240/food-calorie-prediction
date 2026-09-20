"""
NLP Parser module for Food Calorie Prediction.
Extracts food items, quantities, units, and cooking methods from natural language meal descriptions.
"""

import re
from database import get_all_foods, get_food_by_name

# Number words mapping
WORD_NUMBERS = {
    "a": 1.0, "an": 1.0, "one": 1.0, "two": 2.0, "three": 3.0, "four": 4.0,
    "five": 5.0, "six": 6.0, "seven": 7.0, "eight": 8.0, "nine": 9.0, "ten": 10.0,
    "half": 0.5, "quarter": 0.25
}

# Unit multipliers to grams
UNIT_CONVERSIONS = {
    "g": 1.0,
    "gram": 1.0,
    "grams": 1.0,
    "kg": 1000.0,
    "kilogram": 1000.0,
    "kilograms": 1000.0,
    "oz": 28.35,
    "ounce": 28.35,
    "ounces": 28.35,
    "lb": 453.59,
    "pound": 453.59,
    "pounds": 453.59,
    "ml": 1.0,
    "milliliter": 1.0,
    "milliliters": 1.0,
    "l": 1000.0,
    "liter": 1000.0,
    "liters": 1000.0,
    "cup": 240.0, # default liquid/volume
    "cups": 240.0,
    "tbsp": 15.0,
    "tablespoon": 15.0,
    "tablespoons": 15.0,
    "tsp": 5.0,
    "teaspoon": 5.0,
    "teaspoons": 5.0,
    "slice": 40.0,
    "slices": 40.0,
    "piece": 100.0,
    "pieces": 100.0,
    "handful": 30.0,
    "can": 355.0,
    "glass": 250.0,
    "bowl": 250.0
}

COOKING_KEYWORDS = ["raw", "boiled", "steamed", "grilled", "baked", "fried", "deep_fried", "deep fried"]


def parse_fraction(text):
    """Converts strings like '1/2', '1 1/2', or '0.5' to float."""
    text = text.strip()
    if "/" in text:
        parts = text.split()
        if len(parts) == 2:
            return float(parts[0]) + parse_fraction(parts[1])
        num, den = text.split("/")
        return float(num) / float(den)
    return float(text)


def split_meal_text(text):
    """Splits full meal text into individual candidate ingredient phrases."""
    # Replace separators with commas
    normalized = re.sub(r"\b(and|with|\+|plus|\&)\b", ",", text, flags=re.IGNORECASE)
    items = [line.strip() for line in re.split(r"[,;\n\r]+", normalized) if line.strip()]
    return items


def extract_quantity_and_unit(phrase):
    """
    Extracts numeric quantity, unit string, and remaining text.
    Examples:
    '150g grilled chicken' -> (150.0, 'g', 'grilled chicken')
    '2 cups of milk' -> (2.0, 'cups', 'milk')
    '2 eggs' -> (2.0, 'piece', 'eggs')
    '1/2 avocado' -> (0.5, 'piece', 'avocado')
    'an apple' -> (1.0, 'piece', 'apple')
    """
    cleaned = phrase.strip()

    # Pattern 1: Number attached directly to unit like '150g', '200ml', '2oz'
    attached_match = re.search(r"^(\d+(?:\.\d+)?)\s*(g|kg|ml|oz|lb|cups?|tbsp|tsp)\b", cleaned, re.IGNORECASE)
    if attached_match:
        qty = float(attached_match.group(1))
        unit_str = attached_match.group(2).lower()
        remaining = cleaned[attached_match.end():].strip()
        return qty, unit_str, remaining

    # Pattern 2: Fractions like '1 1/2' or '3/4'
    frac_match = re.search(r"^(\d+\s+\d+/\d+|\d+/\d+)", cleaned)
    if frac_match:
        qty = parse_fraction(frac_match.group(1))
        rem = cleaned[frac_match.end():].strip()
        words = rem.split()
        if words and words[0].lower() in UNIT_CONVERSIONS:
            return qty, words[0].lower(), " ".join(words[1:])
        return qty, "piece", rem

    # Pattern 3: Standard decimal or integer like '2' or '2.5'
    num_match = re.search(r"^(\d+(?:\.\d+)?)", cleaned)
    if num_match:
        qty = float(num_match.group(1))
        rem = cleaned[num_match.end():].strip()
        words = rem.split()
        if words and words[0].lower() in UNIT_CONVERSIONS:
            return qty, words[0].lower(), " ".join(words[1:])
        return qty, "piece", rem

    # Pattern 4: Word numbers: 'two eggs', 'a banana'
    first_word = cleaned.split()[0].lower() if cleaned.split() else ""
    if first_word in WORD_NUMBERS:
        qty = WORD_NUMBERS[first_word]
        rem = " ".join(cleaned.split()[1:]).strip()
        words = rem.split()
        if words and words[0].lower() in UNIT_CONVERSIONS:
            return qty, words[0].lower(), " ".join(words[1:])
        return qty, "piece", rem

    # Default fallback
    return 1.0, "serving", cleaned


def detect_cooking_method(text):
    """Detects cooking method from text."""
    lower = text.lower()
    if "deep fried" in lower or "deep-fried" in lower:
        return "deep_fried"
    for m in COOKING_KEYWORDS:
        if m in lower:
            return m
    return "raw"


SYNONYMS = {
    "egg": "Whole Egg",
    "eggs": "Whole Egg",
    "boiled egg": "Whole Egg",
    "boiled eggs": "Whole Egg",
    "chicken": "Chicken Breast (Cooked, Grilled)",
    "chicken breast": "Chicken Breast (Cooked, Grilled)",
    "oats": "Rolled Oats (Dry)",
    "oatmeal": "Rolled Oats (Dry)",
    "rice": "White Rice (Cooked)",
    "white rice": "White Rice (Cooked)",
    "brown rice": "Brown Rice (Cooked)",
    "bread": "Whole Wheat Bread",
    "milk": "Whole Milk (3.25%)",
    "yogurt": "Greek Yogurt (Plain 0% fat)",
    "peanut butter": "Peanut Butter",
    "pizza": "Pepperoni Pizza",
    "fries": "French Fries",
    "burger": "Cheeseburger",
    "pasta": "Pasta (Cooked)",
    "steak": "Lean Beef Sirloin (Cooked)",
    "beef": "Ground Beef (85/15, Cooked)",
    "salad": "Caesar Salad with Chicken",
}


def match_food_item(text, all_foods):
    """
    Matches the remaining ingredient description to the closest known food in our database.
    """
    clean_text = re.sub(r"\b(of|some|fresh|slices|sliced|diced|cooked|piece|pieces|chopped|raw|hot|cold)\b", "", text, flags=re.IGNORECASE)
    clean_text = re.sub(r"\s+", " ", clean_text).strip().lower()

    if not clean_text:
        return None

    # Check direct synonyms first
    if clean_text in SYNONYMS:
        target_name = SYNONYMS[clean_text]
        for food in all_foods:
            if food["name"].lower() == target_name.lower():
                return food

    # Singularize if ends with s (e.g. 'bananas' -> 'banana', 'apples' -> 'apple')
    singular_text = clean_text[:-1] if clean_text.endswith("s") and len(clean_text) > 3 else clean_text
    if singular_text in SYNONYMS:
        target_name = SYNONYMS[singular_text]
        for food in all_foods:
            if food["name"].lower() == target_name.lower():
                return food

    # Exact match
    for food in all_foods:
        if clean_text == food["name"].lower() or singular_text == food["name"].lower():
            return food

    # Substring match
    best_match = None
    best_score = 0
    clean_tokens = set(clean_text.split()).union(set(singular_text.split()))

    for food in all_foods:
        food_name = food["name"].lower()
        base_name = re.sub(r"\(.*?\)", "", food_name).strip()
        food_tokens = set(base_name.split())

        overlap = len(clean_tokens.intersection(food_tokens))
        if overlap > best_score:
            best_score = overlap
            best_match = food
        elif overlap == best_score and overlap > 0 and best_match is not None:
            if abs(len(food_name) - len(clean_text)) < abs(len(best_match["name"]) - len(clean_text)):
                best_match = food

    if best_score > 0:
        return best_match

    # Fallback to checking if any token is inside food name
    for food in all_foods:
        food_name = food["name"].lower()
        for t in clean_tokens:
            if len(t) >= 4 and t in food_name:
                return food

    return None


def parse_meal_description(meal_text, ml_pipeline=None):
    """
    Parses a full meal description into structured items,
    computing weights, macronutrients, and calories (via ML & Atwater).
    """
    all_foods = get_all_foods()
    lines = split_meal_text(meal_text)

    parsed_items = []
    total_calories = 0.0
    total_protein = 0.0
    total_carbs = 0.0
    total_fat = 0.0
    total_fiber = 0.0

    for line in lines:
        qty, unit, remaining = extract_quantity_and_unit(line)
        cooking_method = detect_cooking_method(line)

        # Match to database item
        matched_food = match_food_item(remaining, all_foods)

        if matched_food:
            food_name = matched_food["name"]
            category = matched_food["category"]

            # Calculate total weight in grams
            if unit in UNIT_CONVERSIONS and unit not in ["piece", "pieces", "slice", "slices", "serving"]:
                weight_g = qty * UNIT_CONVERSIONS[unit]
            elif unit in ["slice", "slices"] and "Bread" in food_name:
                weight_g = qty * 40.0
            else:
                # Use standard item weight from DB
                weight_g = qty * matched_food["standard_weight_g"]

            # Scale nutrients from per 100g base
            ratio = weight_g / 100.0
            p = round(matched_food["protein_g"] * ratio, 1)
            c = round(matched_food["carbs_g"] * ratio, 1)
            f = round(matched_food["fat_g"] * ratio, 1)
            fib = round(matched_food["fiber_g"] * ratio, 1)
            sug = round(matched_food["sugar_g"] * ratio, 1)
            water = matched_food["water_pct"]

            # Predict calories using ML if pipeline provided, else Atwater
            if ml_pipeline is not None:
                import pandas as pd
                input_df = pd.DataFrame([{
                    "category": category,
                    "cooking_method": cooking_method,
                    "serving_weight_g": weight_g,
                    "protein_g": p,
                    "carbs_g": c,
                    "fat_g": f,
                    "fiber_g": fib,
                    "sugar_g": sug,
                    "water_pct": water
                }])
                cal = round(float(ml_pipeline.predict(input_df)[0]), 1)
            else:
                # Atwater formula
                net_carbs = max(0.0, c - fib)
                cal = round((4.0 * p) + (4.0 * net_carbs) + (9.0 * f) + (2.0 * fib), 1)

            cal = max(5.0, cal)

            item_info = {
                "original_phrase": line,
                "matched_food": food_name,
                "category": category,
                "cooking_method": cooking_method,
                "quantity": qty,
                "unit": unit,
                "weight_g": round(weight_g, 1),
                "calories": cal,
                "protein_g": p,
                "carbs_g": c,
                "fat_g": f,
                "fiber_g": fib,
                "status": "matched"
            }
        else:
            # Fallback heuristic item
            weight_g = qty * UNIT_CONVERSIONS.get(unit, 100.0)
            cal = round(weight_g * 1.5, 1)  # average estimate
            p = round(weight_g * 0.05, 1)
            c = round(weight_g * 0.20, 1)
            f = round(weight_g * 0.05, 1)
            fib = round(weight_g * 0.02, 1)

            item_info = {
                "original_phrase": line,
                "matched_food": remaining.strip().capitalize() or "Custom Item",
                "category": "Generic",
                "cooking_method": cooking_method,
                "quantity": qty,
                "unit": unit,
                "weight_g": round(weight_g, 1),
                "calories": cal,
                "protein_g": p,
                "carbs_g": c,
                "fat_g": f,
                "fiber_g": fib,
                "status": "estimated"
            }

        parsed_items.append(item_info)
        total_calories += item_info["calories"]
        total_protein += item_info["protein_g"]
        total_carbs += item_info["carbs_g"]
        total_fat += item_info["fat_g"]
        total_fiber += item_info["fiber_g"]

    return {
        "items": parsed_items,
        "total_calories": round(total_calories, 1),
        "total_protein": round(total_protein, 1),
        "total_carbs": round(total_carbs, 1),
        "total_fat": round(total_fat, 1),
        "total_fiber": round(total_fiber, 1),
        "item_count": len(parsed_items)
    }
