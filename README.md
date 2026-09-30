# E-commerce Content-Based Recommender System

A Python implementation of a **Content-Based Recommender System** for e-commerce products.

The project uses product descriptions to identify and recommend items that are similar to a selected product.

## Project Overview

Content-based recommendation focuses on the characteristics of items.

In this project, product descriptions are converted into numerical representations using **TF-IDF (Term Frequency-Inverse Document Frequency)**. The similarity between products is then calculated using **Cosine Similarity**.

The system recommends products with descriptions that are most similar to the selected item.

## Recommendation Process

The recommendation pipeline is:

```text
Product Descriptions
        ↓
TF-IDF Vectorization
        ↓
TF-IDF Matrix
        ↓
Cosine Similarity
        ↓
Similar Products
        ↓
Top-N Recommendations
```

## Libraries

* **Pandas**
* **Scikit-learn**

Main Scikit-learn components:

* `TfidfVectorizer`
* `linear_kernel`

## Dataset

The project uses a sample e-commerce dataset containing product IDs and textual product descriptions.

Dataset file:

`sample-data.csv`

The dataset is included in the `data/` directory.

## Method

### 1. TF-IDF Vectorization

TF-IDF is used to transform product descriptions into numerical vectors.

The vectorizer considers word-level n-grams and removes common English stop words.

### 2. Cosine Similarity

Cosine similarity is calculated between the TF-IDF vectors of all products.

A higher similarity score indicates that two product descriptions are more similar based on their textual content.

### 3. Recommendation

For a selected product, the system:

1. Finds its similarity scores with other products.
2. Sorts the products by similarity.
3. Selects the most similar products.
4. Returns the requested number of recommendations.

## Example

For example, the system can generate recommendations similar to a selected product:

```text
Recommending 4 products similar to Cap 1 graphic t...
------------------------------------------------------------
Recommended: ...
Recommended: ...
Recommended: ...
Recommended: ...
```

The similarity score is calculated using cosine similarity.

## Project Structure

```text
ecommerce_recommender/
│
├── README.md
├── EcommerceRecommender.py
│
├── data/
│   └── sample-data.csv
│
└── docs/
    └── Ecommerce_Content_Based_Recommender.pdf
```

## Documentation

A detailed PDF document describing the project and recommendation approach is available here:

[📄 Ecommerce Content-Based Recommender – PDF](docs/Ecommerce_Content_Based_Recommender.pdf)

## References

* Fritz AI — Recommender Systems
* Dataset source referenced in the original project documentation

## Purpose

This project was created as a practical exercise to understand **Content-Based Recommendation**, **TF-IDF**, and **Cosine Similarity** using Python and Scikit-learn.
