
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


from services.reranker import BM25Reranker
from models.llm_handler import AzureOpenAILLM, OpenAILLM

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
CONFIG['top_k_chunks'] = CONFIG.get('top_k_chunks', 5)

class VectorStoreBase:
    def upload_documents(self, documents: List[dict]):
        raise NotImplementedError

    def search_similar(self, query: str, k: int) -> List[dict]:
        raise NotImplementedError

class ChromaStore(VectorStoreBase):
    def __init__(self):
        persist_dir = CONFIG['chroma_persist_directory']
        os.makedirs(persist_dir, exist_ok=True)

        try:
            self.client = chromadb.PersistentClient(
                path=persist_dir,
                settings=Settings(
                    anonymized_telemetry=False,
                    is_persistent=True
                )
            )
            self.collection = self.client.get_or_create_collection(
                name="rag_documents",
                metadata={"hnsw:space": "cosine"}
            )
        except Exception as e:
            st.error(f"Error initializing ChromaDB: {str(e)}")
            raise

    def get_processed_documents(self) -> set:
        """Get list of already processed document names"""
        try:
            results = self.collection.get()
            if results and results['metadatas']:
                return {meta.get('document_name', '') for meta in results['metadatas']}
            return set()
        except Exception:
            return set()

    def search_similar(self, query_vector: List[float], k: int) -> List[dict]:
        try:
            results = self.collection.query(
                query_embeddings=[query_vector],
                n_results=k,
                include=["documents", "metadatas", "distances"]
            )

            # # Debug print
            # if results and results['metadatas'] and len(results['metadatas']) > 0:
            #     st.write("Debug - First result metadata:", results['metadatas'][0][0])

            similar_chunks = []
            if results and results['documents'] and len(results['documents']) > 0:
                for i in range(len(results['documents'][0])):
                    metadata = results['metadatas'][0][i]

                    # Get metadata values with explicit type conversion
                    source = metadata.get("source", "Unknown")
                    page_num = metadata.get("page_number", "Unknown")
                    chunk_in_page = int(metadata.get("chunk_number_in_page", 0))  # Convert to int
                    total_chunk = int(metadata.get("total_chunk_number", 0))      # Convert to int

                    # Construct the document name without extension
                    doc_name = source.rsplit('.', 1)[0]

                    # Construct the paths
                    pdf_path = os.path.join("pdf_pages", doc_name, f"page_{page_num}.pdf")
                    img_path = os.path.join("pdf_images", doc_name, f"page_{page_num}.png")

                    similar_chunks.append({
                        "content": results['documents'][0][i],
                        "page_number": page_num,
                        "chunk_number_in_page": chunk_in_page,
                        "total_chunk_number": total_chunk,
                        "source": source,
                        "pdf_path": pdf_path,
                        "image_path": img_path,
                        "source_path": metadata.get("source_path", "")
                    })
            return similar_chunks
        except Exception as e:
            st.error(f"Error in search_similar: {str(e)}")
            raise

    def upload_documents(self, documents: List[dict]):
        try:
            ids = [doc["id"] for doc in documents]
            contents = [doc["content"] for doc in documents]
            embeddings = [doc["contentVector"] for doc in documents]
            metadatas = [{
                "page_number": doc["page_number"],
                "source": doc["source"],
                "source_path": doc.get("source_path", ""),
                "document_name": doc["document_name"],
                "chunk_number_in_page": doc["chunk_number_in_page"],
                "total_chunk_number": doc["total_chunk_number"]
            } for doc in documents]

            # Debug print
            st.write("Debug - First document metadata:", metadatas[0])

            batch_size = 100
            for i in range(0, len(documents), batch_size):
                batch_end = min(i + batch_size, len(documents))
                self.collection.add(
                    ids=ids[i:batch_end],
                    documents=contents[i:batch_end],
                    embeddings=embeddings[i:batch_end],
                    metadatas=metadatas[i:batch_end]
                )
        except Exception as e:
            st.error(f"Error in upload_documents: {str(e)}")
            raise

class AzureSearchStore(VectorStoreBase):
    def __init__(self):
        self.search_client = SearchClient(
            endpoint=CONFIG['azure_search_endpoint'],
            index_name=CONFIG['azure_search_index_name'],
            credential=AzureKeyCredential(CONFIG['azure_search_key'])
        )

    def upload_documents(self, documents: List[dict]):
        self.search_client.upload_documents(documents=documents)

    def search_similar(self, query_vector: List[float], k: int) -> List[dict]:
        results = self.search_client.search(
            search_text=None,
            vector=query_vector,
            top_k=k,
            vector_fields="contentVector",
            select=["content", "page_number", "source", "pdf_path", "image_path"]
        )
        return [dict(result) for result in results]