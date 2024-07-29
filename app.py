import streamlit as st
from src.data_collection import fetch_video_data, download_video, capture_frame_from_video
from src.feature_extraction import FeatureExtractor
import os

def fetch_and_process_video(api_key, query):
    """
    Fetches video data, captures a frame, and extracts features.
    
    Parameters:
        api_key (str): YouTube API key.
        query (str): Search query.
        
    Returns:
        list: List of recommended video metadata.
    """
    video_data = fetch_video_data(api_key, query)
    
    # Debugging: Print the API response
    st.write("API Response:")
    st.write(video_data)
    
    if 'items' not in video_data:
        st.error("No items found in the API response.")
        return []
    
    recommendations = []
    
    for item in video_data['items']:
        video_id = item['id']['videoId']
        video_url = f"https://www.youtube.com/watch?v={video_id}"
        title = item['snippet']['title']
        thumbnail_url = item['snippet']['thumbnails']['high']['url']
        
        try:
            video_file = download_video(video_url)
            frame = capture_frame_from_video(video_file, timestamp_sec=10)  # Capture frame at 10 seconds
            os.remove(video_file)  # Clean up video file after extraction
        except ValueError as e:
            print(f"Error: {e}")
            continue
        
        # Extract features (optional for further processing)
        extractor = FeatureExtractor()
        features = extractor.extract_features(frame)
        
        recommendations.append({
            'title': title,
            'url': video_url,
            'thumbnail': thumbnail_url
        })
    
    return recommendations

def main():
    st.title("YouTube Video Recommendation System")
    
    # Print available secrets for debugging
    st.write("Available secrets:")
    st.write(st.secrets)
    
    # Fetch API key from secrets
    try:
        api_key = st.secrets["youtube_api_key"]["youtube_api_key"]
        st.write(f"API Key: {api_key}")
    except KeyError as e:
        st.error(f"KeyError: {str(e)}")
    
    # Input for search query
    query = st.text_input("Enter search keyword:")
    
    if query:
        recommendations = fetch_and_process_video(api_key, query)
        
        if recommendations:
            st.write("Recommended Videos:")
            for video in recommendations:
                st.image(video['thumbnail'], width=200)
                st.write(f"[{video['title']}]({video['url']})")
        else:
            st.write("No recommendations found.")

if __name__ == "__main__":
    main()
