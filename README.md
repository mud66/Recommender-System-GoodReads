# Recommender-System-GoodReads
******In Progress*****

Repository for undergrad thesis at University of York, graduating 2025

Using content and collaborative based filtering build a recommender system for the goodreads dataset

- dont include sequels of the book or movie
- how to properly evaluate a recommendation
- gridsearchcv
- hybrid approach

One important one to consider is the source of user preference, is it explicit (thumbs up, stars, ratings) or implicit (views, clicks, time spent, purchases).

1. Weighted Hybrid
In this approach, you generate recommendations using both content-based and collaborative filtering methods, then combine the results by assigning weights to each method. For example:

content_score = content_based_model.predict(user, item)

collab_score = collaborative_model.predict(user, item)

final_score = 0.5 * content_score + 0.5 * collab_score

You can adjust the weights based on the performance of each model.

2. Switching Hybrid
This method switches between content-based and collaborative filtering based on certain conditions. For example, use content-based filtering for new users (cold start problem) and collaborative filtering for users with sufficient interaction history.

3. Feature Augmentation
Use the output of one model as an input feature for the other. For example, use collaborative filtering to generate user preferences and include these as features in the content-based model.

4. Model Blending
Train separate models for content-based and collaborative filtering, then blend their outputs using machine learning techniques. For example, you can use a meta-learner to combine the predictions from both models.

Benefits of Hybrid Systems:
Improved Accuracy: Leverages the strengths of both methods.
Diverse Recommendations: Provides a broader range of recommendations.
Cold Start Problem: Mitigates issues with new users or items.
Hybrid systems are widely used in industry, including by companies like Netflix and Amazon, to enhance recommendation accuracy and user satisfaction
