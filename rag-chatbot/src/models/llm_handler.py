# Import statements
import os
from typing import List, Dict, Tuple
import streamlit as st
from datetime import datetime
import chromadb
from chromadb.config import Settings
import yaml
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
from langchain_community.embeddings import OpenAIEmbeddings
from azure.search.documents import SearchClient
from azure.core.credentials import AzureKeyCredential
from openai import AzureOpenAI, OpenAI
import numpy as np
import json
from PyPDF2 import PdfReader, PdfWriter
import base64
import pdf2image
from PIL import Image
import io
import time
from dotenv import load_dotenv


def load_config():
    """Load configuration from yaml and environment variables"""
    # Load OpenAI API key from .env
    load_dotenv()

    # Load other configurations from yaml
    with open('config.yaml', 'r') as file:
        config = yaml.safe_load(file)

    # Override OpenAI API key from environment variable
    config['openai_api_key'] = os.getenv('OPENAI_API_KEY')

    # Set default for top_k_chunks if not in yaml
    config['top_k_chunks'] = config.get('top_k_chunks', 5)

    return config

CONFIG = load_config()
CONFIG['top_k_chunks'] = CONFIG.get('top_k_chunks', 5)  # Default to 5 if not specified

class LLMBase:
    def generate_completion(self, messages: List[dict]) -> str:
        raise NotImplementedError

    def generate_completion_with_images(self, messages: List[dict], image_paths: List[str]) -> str:
        raise NotImplementedError

class AzureOpenAILLM(LLMBase):
    def __init__(self):
        self.client = AzureOpenAI(
            api_key=CONFIG['azure_openai_key'],
            api_version="2023-05-15",
            azure_endpoint=CONFIG['azure_openai_endpoint']
        )

    def generate_completion(self, messages: List[dict]) -> str:
        response = self.client.chat.completions.create(
            model=CONFIG['azure_deployment_name'],
            messages=messages,
            temperature=0.1,
            max_tokens=300,
            presence_penalty=0.0,
            frequency_penalty=0.0
        )
        return response.choices[0].message.content

    def generate_completion_with_images(self, messages: List[dict], image_paths: List[str]) -> str:
        try:
            # Prepare the content list with the text prompt first
            content_list = [{"type": "text", "text": messages[-1]["content"]}]

            # Add each image to the content list
            for img_path in image_paths:
                with open(img_path, "rb") as image_file:
                    base64_image = base64.b64encode(image_file.read()).decode('utf-8')
                    content_list.append({
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{base64_image}"
                        }
                    })

            # Construct the final messages array
            final_messages = [
                messages[0],  # Keep the system message
                {
                    "role": "user",
                    "content": content_list
                }
            ]

            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=final_messages,
                temperature=0.1,
                max_tokens=300
            )
            return response.choices[0].message.content

        except Exception as e:
            st.error(f"Error in image processing: {str(e)}")
            raise

class OpenAILLM(LLMBase):
    def __init__(self):
        self.client = OpenAI(api_key=CONFIG['openai_api_key'])

    def generate_completion(self, messages: List[dict]) -> str:
        response = self.client.chat.completions.create(
            model=CONFIG['model_name'],
            messages=messages,
            temperature=0.1,
            max_tokens=300,
            presence_penalty=0.0,
            frequency_penalty=0.0
        )
        return response.choices[0].message.content

    def is_valid_image(self, image_path: str, max_size_mb: float = 1.0, allowed_formats: List[str] = ["png", "jpeg", "jpg"]) -> Tuple[bool, str]:
        """
        Validate the image file.

        Args:
            image_path (str): Path to the image file.
            max_size_mb (float): Maximum allowed size in MB.
            allowed_formats (List[str]): List of allowed image formats.

        Returns:
            Tuple[bool, str]: A tuple containing a boolean indicating if the image is valid and an error message if invalid.
        """
        try:
            # Check file extension
            ext = image_path.split('.')[-1].lower()
            if ext not in allowed_formats:
                return False, f"Unsupported image format '{ext}'. Allowed formats: {', '.join(allowed_formats)}."

            # Check file size
            file_size_mb = os.path.getsize(image_path) / (1024 * 1024)
            if file_size_mb > max_size_mb:
                return False, f"Image size exceeds the limit of {max_size_mb} MB (current size: {file_size_mb:.2f} MB)."

            # Attempt to open the image
            with Image.open(image_path) as img:
                img.verify()  # Check if it's a valid image file
            return True, ""
        except Exception as e:
            return False, f"Error validating image: {str(e)}"


    def generate_completion_with_images(self, messages: List[dict], image_paths: List[str]) -> str:
        try:
            # Prepare the content list with the text prompt
            content_list = [{"type": "text", "text": messages[-1]["content"]}]

            # Track already added image URLs to prevent duplicates
            added_images = set()

            # Validate and add images
            for img_path in image_paths:
                is_valid, error_message = self.is_valid_image(img_path)
                if not is_valid:
                    st.warning(f"Skipping image '{img_path}': {error_message}")
                    continue

                with open(img_path, "rb") as image_file:
                    base64_image = base64.b64encode(image_file.read()).decode("utf-8")
                    image_url = f"data:image/png;base64,{base64_image}"

                    # Check for duplicates
                    if image_url in added_images:
                        st.warning(f"Skipping duplicate image: {img_path}")
                        continue

                    # Add unique image URL
                    content_list.append({
                        "type": "image_url",
                        "image_url": {"url": image_url}
                    })
                    added_images.add(image_url)  # Track this image as added

            if len(content_list) == 1:
                raise ValueError("No valid images were added to the request.")

            # Construct the final messages array
            final_messages = [
                messages[0],  # Keep the system message
                {
                    "role": "user",
                    "content": content_list
                }
            ]

            # Debug: Inspect the final messages structure
            st.write("Debug - Final messages sent to API:", final_messages)

            # Make the API call
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=final_messages,
                temperature=0.1,
                max_tokens=300
            )

            return response.choices[0].message.content

        except Exception as e:
            st.error(f"Error in image processing: {str(e)}")
            raise
