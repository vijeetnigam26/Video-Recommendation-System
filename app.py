import streamlit as st
import requests
import pandas as pd
from io import BytesIO
from pytube import YouTube
from PIL import Image

# Define a function to fetch and process YouTube videos based on the search keyword
def fetch_and_process_video(api_key, search_keyword):
    # YouTube API endpoint
    youtube_api_url = f"https://www.googleapis.com/youtube/v3/search?part=snippet&q={search_keyword}&key={api_key}&type=video&maxResults=5"

    # Send request to YouTube API
    response = requests.get(youtube_api_url)
    data = response.json()

    # Check if the response contains items
    if 'items' not in data:
        st.error("Error: No items found in the API response.")
        return None

    videos = []

    for item in data['items']:
        video_id = item['id']['videoId']
        title = item['snippet']['title']
        thumbnail_url = item['snippet']['thumbnails']['high']['url']
        video_url = f"https://www.youtube.com/watch?v={video_id}"

        # Download the video thumbnail
        try:
            thumbnail_response = requests.get(thumbnail_url)
            thumbnail_image = Image.open(BytesIO(thumbnail_response.content))
            video_file = {
                'title': title,
                'video_url': video_url,
                'thumbnail': thumbnail_image
            }
            videos.append(video_file)
        except Exception as e:
            st.error(f"Error processing video thumbnail: {e}")

    return videos

# Streamlit app
def main():
    st.title("YouTube Video Recommendation System")

    # Input field for search keyword
    api_key = st.secrets["youtube"]["api_key"]
    search_keyword = st.text_input("Enter search keyword:", "")

    if st.button("Search"):
        if search_keyword:
            with st.spinner("Fetching video recommendations..."):
                videos = fetch_and_process_video(api_key, search_keyword)
                
                if videos:
                    for video in videos:
                        st.subheader(video['title'])
                        st.image(video['thumbnail'], use_column_width=True)
                        st.write(f"[Watch Video]({video['video_url']})")
        else:
            st.error("Please enter a search keyword.")

if __name__ == "__main__":
    main()