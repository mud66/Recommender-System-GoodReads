def rescale_predictions(predicted_ratings, user_id, user_bias, item_bias, global_mean):
    rescaled_ratings = []
    for book_id, normalized_rating in predicted_ratings:
        user_b = user_bias.get(user_id, 0)
        item_b = item_bias.get(book_id, 0)
        original_rating = normalized_rating + user_b + item_b + global_mean
        rescaled_ratings.append((book_id, original_rating))
    return rescaled_ratings
