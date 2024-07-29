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
    """
    Downloads the video from YouTube.
    
    Parameters:
        video_url (str): URL of the YouTube video.
        
    Returns:
        str: File path of the downloaded video.
    """
    yt = YouTube(video_url)
    stream = yt.streams.filter(file_extension='mp4').first()
    video_file = 'downloaded_video.mp4'
    stream.download(filename=video_file)
    return video_file

def capture_frame_from_video(video_file, timestamp_sec):
    """
    Captures a frame from a video at a specific timestamp.
    
    Parameters:
        video_file (str): File path of the video.
        timestamp_sec (int): Timestamp in seconds to capture the frame.
        
    Returns:
        np.ndarray: Captured frame as an image.
    """
    video_capture = cv2.VideoCapture(video_file)
    fps = video_capture.get(cv2.CAP_PROP_FPS)
    frame_number = int(fps * timestamp_sec)
    
    # Set frame position
    video_capture.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    success, frame = video_capture.read()
    video_capture.release()
    
    if success:
        return frame
    else:
        raise ValueError(f"Unable to capture frame at timestamp {timestamp_sec} seconds.")