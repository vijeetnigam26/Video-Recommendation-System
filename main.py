import numpy as np
from data_collection import fetch_video_data, fetch_user_interactions, save_frame_to_file, capture_frame_from_video
from feature_extraction import FeatureExtractor
from recommendation import recommend_videos, collaborative_filtering
from get_api import get_youtube_api_key
from user_behavior import user_behavior_analysis

def extract_video_id_from_url(url):
    """
    Extracts the video ID from a YouTube video URL.
    
    Parameters:
        url (str): YouTube video URL.
        
    Returns:
        str: Extracted video ID.
    """
    return url.split("v=")[-1]

def main():
    """
    Main function to execute the video recommendation system workflow.
    """
    api_key = get_youtube_api_key()
    video_url = 'https://www.youtube.com/watch?v=YOUR_VIDEO_ID'
    video_id = extract_video_id_from_url(video_url)
    
    # Fetch and save video data
    video_data = fetch_video_data(api_key, video_id)
    frame = capture_frame_from_video(video_url, timestamp_sec=10)  # Capture frame at 10 seconds
    save_frame_to_file(frame)
    
    # Fetch user interactions
    user_interactions = fetch_user_interactions(api_key, video_id)
    
    # Extract features from the captured frame
    extractor = FeatureExtractor()
    features = extractor.extract_features('frame.jpg')
    
    # Simulate video features and user interaction data
    other_video_features = np.random.rand(100, 512)  # Simulated features for other videos
    user_interaction_data = np.random.rand(100, 10)  # Simulated user interaction matrix
    
    # Generate recommendations
    content_based_recommendations = recommend_videos(features, other_video_features)
    collaborative_recommendations = collaborative_filtering(user_interaction_data)
    
    # Perform user behavior analysis
    analysis_results = user_behavior_analysis(user_interaction_data, num_clusters=5)
    
    # Print results
    print("Content-based recommendations:", content_based_recommendations)
    print("Collaborative filtering recommendations:", collaborative_recommendations)
    print("User Clusters:", analysis_results['user_clusters'])
    print("Cluster Centers:", analysis_results['cluster_centers'])
    print("Scaled Data:", analysis_results['scaled_data'])

if __name__ == "__main__":
    main()