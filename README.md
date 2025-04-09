# 📚 Recommender-System-GoodReads  
**Undergraduate Thesis – University of York (2025)**  
**Dataset**: [Goodreads](https://mengtingwan.github.io/data/goodreads.html#datasets)

---

## Project Overview  
Current recommendation methods often favour bestsellers and popular titles, leaving niche or lesser-known books overlooked. Goodreads, while widely used, fails to offer truly personalised and transparent suggestions due to its outdated design and popularity-based algorithms. This project addresses these limitations by building a hybrid recommendation system that combines collaborative filtering, content-based methods, and graph-based models to enhance:  
- **Accuracy**, even with sparse or cold-start data  
- **Novelty**, helping users discover overlooked books  
- **Explainability**, offering insights into why a book was recommended  

---
## Core Models and Techniques  

| Model        | Description |
|-------------|-------------|
| **SVD**      | Used to impute ratings for books that users have read but not rated. These imputed ratings are used by NMF and GATv2Conv. |
| **NMF**      | Matrix factorisation model for predicting ratings and offering interpretability via latent factor analysis. |
| **GATv2Conv**| A graph-based model using review sentiment, review embeddings, book genres, and user genre preferences to predict ratings. |
| **HDBSCAN**  | Clustering books based on content feature embeddings for content-based recommendations, especially for cold-start and new users. |

---
No data processing required — preprocessed files and trained models are saved in the `Pickle` folder.  

- **LoadData**: Sample and merge datasets to align IDs, expand shelves, add genres, and save processed data.  
- **EDA**: Exploratory plots and dataset statistics.
- **Book Embeddings**: Uses `SentenceTransformer('all-MiniLM-L6-v2')` on combined book metadata (title, description, authors, genres, shelves).  
- **Review Embeddings**: Same model, applied to review text. Saved to `Pickle/review_embeddings.pkl`.  
- **Sentiment Scores**: Uses `"distilbert-base-uncased-finetuned-sst-2-english"` via HuggingFace pipeline to assign sentiment scores and confidence. Saved to `Pickle/review_score.pkl`.  
- **User Genres**: Extracts top 4 genres read by each user. Saved to `Pickle/user_most_common_genres.pkl`.  

##  Recommender Models  

###  SVD  
- Used on user-book interactions with missing ratings filtered and preprocessed.  
- 2% of original rows reintroduced for variety.  
- Ratings predicted for missing entries and saved to `Pickle/best_svd_model.pkl`.  
- If a user/book is missing, rating left null (no fallback to global mean).  

###  NMF  
- Dataset split, balanced (rating classes ≥0.75% of majority), and normalised.  
- Uses SVD-imputed ratings where user has read but not rated a book.  
- Trained using `Surprise` with hyperparameter tuning (GridSearchCV, RMSE minimised).  
- Returns top 5 books with predicted ratings and explanation from latent factor contributions.

###  GATv2Conv  
- Input data includes reviews, sentiment scores, embeddings, user genres, book genres, and imputed ratings.  
- Dataset split 80/10/10 and upsampled (minority classes to 75%).  
- Node features: one-hot encoded genres.  
- Edge attributes: ratings, sentiment scores, review embeddings.  
- Architecture:
  - 5 GATv2Conv layers  
  - 30 hidden channels  
  - 25 attention heads  
  - ELU activation, dropout 0.2, AdamW (lr=1e-5, weight decay=1e-4)  
  - MSE loss, early stopping (patience=10)  
- Recommender calculates predicted ratings (dot product of user/book embeddings) and returns top 5 results.

###  HDBSCAN  
- Book feature embeddings reduced via UMAP (10 components, cosine distance).  
- HDBSCAN applies soft clustering with Euclidean distance.  
- Membership vectors used to recommend thematically similar books.  
- For low-interaction users:
  - Recommend from shared clusters (≥0.01 probability), else FAISS-based fallback.  
- For new users:
  - Recommend from sampled clusters, ranked by L2 norm (proximity to origin), with fallback to global similarity or random.  

---

##  Hybrid System  

![Hybrid Pipeline](pipeline%20(1).png)
Main hybrid pipeline in 'Hybrid/Hybrid.ipynb' with gat .py file to load model.

## References
Mengting Wan, Julian McAuley, "Item Recommendation on Monotonic Behavior Chains", in RecSys'18.
Mengting Wan, Rishabh Misra, Ndapa Nakashole, Julian McAuley, "Fine-Grained Spoiler Detection from Large-Scale Review Corpora", in ACL'19.
