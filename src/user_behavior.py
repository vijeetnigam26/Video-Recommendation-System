import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

def user_behavior_analysis(user_interaction_data, num_clusters=5):
    """
    Analyzes user behavior using clustering.
    
    Parameters:
        user_interaction_data (numpy.ndarray): User interaction matrix.
        num_clusters (int): Number of clusters for KMeans algorithm.
        
    Returns:
        dict: User clusters and cluster centers.
    """
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(user_interaction_data)
    
    kmeans = KMeans(n_clusters=num_clusters, random_state=0)
    clusters = kmeans.fit_predict(scaled_data)
    cluster_centers = kmeans.cluster_centers_
    
    results = {
        'user_clusters': clusters,
        'cluster_centers': cluster_centers,
        'scaled_data': scaled_data
    }
    
    return results