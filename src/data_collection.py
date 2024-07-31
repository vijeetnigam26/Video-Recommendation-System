import cv2
import numpy as np
import requests
from pytube import YouTube
from io import BytesIO

import requests

def fetch_video_data(api_key, query):
    """
    Fetches video data from YouTube API based on the search query.
    
    Parameters:
        api_key (str): YouTube API key.
        query (str): Search query.
        
    Returns:
        dict: API response containing video data.
    """
    url = "https://www.googleapis.com/youtube/v3/search"
    params = {
        'part': 'snippet',
        'q': query,
        'type': 'video',
        'key': api_key,
        'maxResults': 5  # Number of results to fetch
    }
    response = requests.get(url, params=params)
    return response.json()

def download_video(video_url):
    try:
        yt = YouTube(video_url)
        stream = yt.streams.get_highest_resolution()
        # Save the video
        video_file = stream.download(filename='video.mp4')
        return video_file
    except requests.exceptions.RequestException as e:
        raise ValueError(f"Error downloading video: {e}")
    except Exception as e:
        raise ValueError(f"Unexpected error: {e}")

def capture_frame_from_video(video_file, timestamp_sec):
    try:
        cap = cv2.VideoCapture(video_file)
        cap.set(cv2.CAP_PROP_POS_MSEC, timestamp_sec * 1000)
        success, frame = cap.read()
        cap.release()
        if not success:
            raise ValueError("Failed to capture frame")
        return frame
    except Exception as e:
        raise ValueError(f"Error capturing frame: {e}")