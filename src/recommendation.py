import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def recommend_videos(features, other_video_features):
    """
    Recommends videos based on content similarity using cosine similarity.
    
    Parameters:
        features (numpy.ndarray): Features of the current video.
        other_video_features (numpy.ndarray): Features of other videos.
        
    Returns:
        list: Indices of recommended videos.
    """
    similarities = cosine_similarity(features.reshape(1, -1), other_video_features)
    recommended_indices = similarities.argsort()[0][-5:]  # Top 5 recommendations
    return recommended_indices.tolist()

def collaborative_filtering(user_interaction_data):
    """
    Recommends videos based on user interaction data using collaborative filtering.
    
    Parameters:
        user_interaction_data (numpy.ndarray): User interaction matrix.
        
    Returns:
        list: Indices of recommended videos.
    """
    similarities = cosine_similarity(user_interaction_data)
    recommended_indices = similarities.argsort()[:, -5:]  # Top 5 recommendations for each item
    return recommended_indices.tolist()