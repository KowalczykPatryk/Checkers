"""
Contains function to download model from google drive and saves it in provided filepath
"""

import gdown

def download_model(url: str, filepath: str):
    """
    Downloads model from google drive and saves it in provided filepath
    """
    gdown.download(url, filepath, quiet=False)
