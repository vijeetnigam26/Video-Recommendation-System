import json

def get_youtube_api_key(config_file='config.json'):
    """
    Retrieves the YouTube API key from the configuration file.
    
    Parameters:
        config_file (str): Path to the configuration file.
        
    Returns:
        str: YouTube API key.
    """
    with open(config_file, 'r') as file:
        config = json.load(file)
    return config.get('youtube_api_key')