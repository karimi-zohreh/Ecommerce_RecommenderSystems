#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel


# Load dataset
data = pd.read_csv("data/sample-data.csv")


# Create TF-IDF matrix
tfidf = TfidfVectorizer(
    analyzer="word",
    ngram_range=(1, 3),
    stop_words="english"
)

tfidf_matrix = tfidf.fit_transform(data["description"])


# Calculate cosine similarity
cosine_similarities = linear_kernel(tfidf_matrix, tfidf_matrix)


# Store similarity results
result = {}

for item_index, row in data.iterrows():
    similar_indices = cosine_similarities[item_index].argsort()[::-1]

    similar_items = [
        (
            cosine_similarities[item_index][i],
            data["id"].iloc[i]
        )
        for i in similar_indices
        if data["id"].iloc[i] != row["id"]
    ]

    result[row["id"]] = similar_items


# Get item description
def get_item_description(item_id):
    return (
        data.loc[data["id"] == item_id, "description"]
        .iloc[0]
        .split("-")[0]
    )


# Generate recommendations
def recommend(item_id, num_recommendations=4):
    item_name = get_item_description(item_id)

    print(
        f"Recommending {num_recommendations} products "
        f"similar to {item_name}..."
    )
    print("-" * 60)

    recommendations = result[item_id][:num_recommendations]

    for score, recommended_id in recommendations:
        print(
            f"Recommended: "
            f"{get_item_description(recommended_id)} "
            f"(score: {score:.4f})"
        )


# Example
recommend(item_id=20, num_recommendations=4)

