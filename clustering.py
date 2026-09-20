"""
Unsupervised Learning & Clustering Module for Food Calorie Prediction.
Clusters food items using K-Means based on macronutrient density and PCA 2D projections
for visual analytics on the dashboard.
"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from database import get_all_foods

CLUSTER_LABELS = {
    0: "Lean Protein Dominant",
    1: "Complex Carbohydrates & Energy",
    2: "Healthy Fats & Caloric Density",
    3: "Micronutrient-Rich & Hydrating (Low-Cal)",
    4: "Processed & High-Density Mixed"
}


def perform_food_clustering(k=5):
    """
    Extracts all foods from the database, normalizes macronutrients per 100g,
    clusters them using KMeans, and reduces dimensions via PCA for visualization.
    """
    foods = get_all_foods()
    if not foods:
        return {"clusters": [], "points": [], "cluster_summaries": []}

    df = pd.DataFrame(foods)

    feature_cols = ["calories", "protein_g", "carbs_g", "fat_g", "fiber_g", "water_pct"]
    X = df[feature_cols].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    cluster_ids = kmeans.fit_predict(X_scaled)

    # 2D PCA for visual representation
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_scaled)

    # Attach cluster info
    df["cluster_id"] = cluster_ids
    df["pca_x"] = np.round(X_pca[:, 0], 3)
    df["pca_y"] = np.round(X_pca[:, 1], 3)

    # Compute cluster summaries
    cluster_summaries = []
    for c_id in range(k):
        sub = df[df["cluster_id"] == c_id]
        if len(sub) == 0:
            continue
        cluster_summaries.append({
            "cluster_id": int(c_id),
            "label": CLUSTER_LABELS.get(c_id, f"Cluster {c_id}"),
            "food_count": int(len(sub)),
            "avg_calories": round(float(sub["calories"].mean()), 1),
            "avg_protein": round(float(sub["protein_g"].mean()), 1),
            "avg_carbs": round(float(sub["carbs_g"].mean()), 1),
            "avg_fat": round(float(sub["fat_g"].mean()), 1),
            "sample_foods": sub["name"].head(4).tolist()
        })

    # Prepare data points for frontend Chart.js scatter plot
    points = []
    for _, row in df.iterrows():
        points.append({
            "name": row["name"],
            "category": row["category"],
            "calories": float(row["calories"]),
            "protein": float(row["protein_g"]),
            "carbs": float(row["carbs_g"]),
            "fat": float(row["fat_g"]),
            "cluster_id": int(row["cluster_id"]),
            "cluster_label": CLUSTER_LABELS.get(row["cluster_id"], f"Cluster {row['cluster_id']}"),
            "x": float(row["pca_x"]),
            "y": float(row["pca_y"])
        })

    return {
        "points": points,
        "cluster_summaries": cluster_summaries,
        "pca_variance_ratio": [round(float(v), 3) for v in pca.explained_variance_ratio_]
    }


if __name__ == "__main__":
    res = perform_food_clustering()
    print(f"Generated {len(res['points'])} clustered points across {len(res['cluster_summaries'])} clusters.")
    for c in res["cluster_summaries"]:
        print(f"- {c['label']}: {c['food_count']} foods (Avg Cal: {c['avg_calories']}) Samples: {c['sample_foods']}")
