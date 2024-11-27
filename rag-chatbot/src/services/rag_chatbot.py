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
from services.vectordb_handler import ChromaStore, AzureSearchStore
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
CONFIG['top_k_chunks'] = CONFIG.get('top_k_chunks', 5)  # Default to 5 if not specified

class ConversationMemory:
    def __init__(self, max_memory: int = CONFIG['max_memory_messages']):
        self.messages = []
        self.max_memory = max_memory

    def add_message(self, role: str, content: str):
        self.messages.append({
            "role": role,
            "content": content,
            "timestamp": datetime.now()
        })
        if len(self.messages) > self.max_memory:
            self.messages = self.messages[-self.max_memory:]

    def get_conversation_context(self) -> str:
        return "\n".join([
            f"{msg['role']}: {msg['content']}"
            for msg in self.messages[-self.max_memory:]
        ])

class RAGChatbot:
    def __init__(self):
        self.MAX_FILE_SIZE = 25 * 1024 * 1024  # 25MB in bytes
        self.PDF_PAGES_DIR = "pdf_pages"
        self.PDF_IMAGES_DIR = "pdf_images"

        os.makedirs(self.PDF_PAGES_DIR, exist_ok=True)
        os.makedirs(self.PDF_IMAGES_DIR, exist_ok=True)

        self.embeddings = OpenAIEmbeddings(
            model=CONFIG['embedding_model_name'],
            openai_api_key=CONFIG['openai_api_key']
        )

        if CONFIG['vector_store'] == 'AZURE_SEARCH':
            self.vector_store = AzureSearchStore()
        else:
            self.vector_store = ChromaStore()

        if CONFIG['llm_type'] == 'AZURE_OPENAI':
            self.llm = AzureOpenAILLM()
        else:
            self.llm = OpenAILLM()

        self.memory = ConversationMemory()
        self.current_similar_chunks = []

        self.reranker = BM25Reranker()

    def validate_file_size(self, file) -> bool:
        return file.size <= self.MAX_FILE_SIZE

    def save_pdf_pages(self, uploaded_file, target_width=800) -> Tuple[str, str]:
        """
        Save PDF pages as images and resize them to reduce file size.

        Args:
            uploaded_file: The uploaded PDF file.
            target_width: The desired width of the resized images.

        Returns:
            Tuple[str, str]: Paths to the directories containing the saved PDF pages and images.
        """
        # Simplify directory name - remove timestamp
        doc_name = uploaded_file.name.rsplit('.', 1)[0]
        pdf_dir = os.path.join(self.PDF_PAGES_DIR, doc_name)  # Simplified path
        img_dir = os.path.join(self.PDF_IMAGES_DIR, doc_name)  # Simplified path

        os.makedirs(pdf_dir, exist_ok=True)
        os.makedirs(img_dir, exist_ok=True)

        temp_path = os.path.join(pdf_dir, "temp.pdf")
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getvalue())

        try:
            images = pdf2image.convert_from_path(temp_path, dpi=150)  # Adjust DPI for initial image quality
            pdf = PdfReader(temp_path)

            for page_num in range(len(pdf.pages)):
                # Save as PDF
                writer = PdfWriter()
                writer.add_page(pdf.pages[page_num])
                pdf_path = os.path.join(pdf_dir, f"page_{page_num + 1}.pdf")
                with open(pdf_path, "wb") as output_file:
                    writer.write(output_file)

                # Save as image
                img_path = os.path.join(img_dir, f"page_{page_num + 1}.png")
                images[page_num].save(img_path, "PNG")

                # Resize the image
                with Image.open(img_path) as img:
                    original_width, original_height = img.size
                    scale_ratio = target_width / original_width
                    new_height = int(original_height * scale_ratio)
                    resized_img = img.resize((target_width, new_height), Image.ANTIALIAS)
                    resized_img.save(img_path, "PNG")  # Overwrite the original image

        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

        return pdf_dir, img_dir


    def batch_embed_texts(self, texts: List[str], batch_size: int = 32) -> List[List[float]]:
        """Embed multiple texts in batches for better performance"""
        embeddings = []
        total = len(texts)
        progress_bar = st.progress(0)

        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            # Process entire batch at once instead of one by one
            batch_embeddings = self.embeddings.embed_documents(batch_texts)
            embeddings.extend(batch_embeddings)

            # Update progress
            progress = min((i + batch_size) / total, 1.0)
            progress_bar.progress(progress)
            if i % batch_size == 0:
                st.write(f"Processed {i}/{total} embeddings...")

        progress_bar.empty()
        return embeddings

    def process_document(self, uploaded_file) -> Tuple[List[dict], str, str]:
        """Process a document, checking first if it's already been processed"""
        if not self.validate_file_size(uploaded_file):
            raise ValueError(f"File size exceeds maximum limit of 25MB. Current size: {uploaded_file.size / (1024 * 1024):.2f}MB")

        # Check if document was already processed
        if isinstance(self.vector_store, ChromaStore):
            processed_docs = self.vector_store.get_processed_documents()
            if uploaded_file.name in processed_docs:
                # Document exists in vector store, check if PDF and image files exist
                doc_name = uploaded_file.name.rsplit('.', 1)[0]
                pdf_dir = os.path.join(self.PDF_PAGES_DIR, doc_name)
                img_dir = os.path.join(self.PDF_IMAGES_DIR, doc_name)

                if os.path.exists(pdf_dir) and os.path.exists(img_dir):
                    # Both vector store entry and files exist, no need to reprocess
                    st.info(f"Document '{uploaded_file.name}' has already been processed and stored.")
                    return [], pdf_dir, img_dir
                else:
                    # Vector store has the document but files are missing
                    # Only regenerate the PDF and image files
                    st.warning(f"Document '{uploaded_file.name}' exists in database but files are missing. Regenerating files...")
                    return self.regenerate_document_files(uploaded_file)

        # Process new document
        doc_name = uploaded_file.name.rsplit('.', 1)[0]
        pdf_dir = os.path.join(self.PDF_PAGES_DIR, doc_name)
        img_dir = os.path.join(self.PDF_IMAGES_DIR, doc_name)

        os.makedirs(pdf_dir, exist_ok=True)
        os.makedirs(img_dir, exist_ok=True)

        # Save the full PDF temporarily
        temp_path = os.path.join(pdf_dir, "temp.pdf")
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getvalue())

        try:
            # Convert PDF to images
            images = pdf2image.convert_from_path(temp_path, dpi=200)
            pdf = PdfReader(temp_path)

            # Save individual pages
            for page_num in range(len(pdf.pages)):
                # Save as PDF
                writer = PdfWriter()
                writer.add_page(pdf.pages[page_num])
                pdf_path = os.path.join(pdf_dir, f"page_{page_num + 1}.pdf")
                with open(pdf_path, "wb") as output_file:
                    writer.write(output_file)

                # Save as image
                img_path = os.path.join(img_dir, f"page_{page_num + 1}.png")
                images[page_num].save(img_path, "PNG")

                # Debug: Print file existence
                st.write(f"Debug - Created files for page {page_num + 1}:")
                st.write(f"PDF exists: {os.path.exists(pdf_path)}")
                st.write(f"Image exists: {os.path.exists(img_path)}")

            # Process text and create chunks
            loader = PyPDFLoader(temp_path)
            pages = loader.load()

            # Use larger chunk size to reduce number of embeddings needed
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1500,  # Increased from 1000
                chunk_overlap=150,  # Increased from 100
                separators=["\n\n", "\n", " ", ""]
            )

            chunks = []
            page_chunk_counters = {}  # Track chunks per page
            total_chunks = 0  # Track total chunks

            with st.spinner("Processing PDF..."):
                for page in pages:
                    splits = text_splitter.split_text(page.page_content)
                    page_num = page.metadata.get("page", 0) + 1

                    # Initialize counter for this page
                    if page_num not in page_chunk_counters:
                        page_chunk_counters[page_num] = 0

                    for split in splits:
                        page_chunk_counters[page_num] += 1
                        total_chunks += 1

                        # Debug print for first chunk
                        if total_chunks == 1:
                            st.write("Debug - First chunk data:", {
                                "page_num": page_num,
                                "chunk_in_page": page_chunk_counters[page_num],
                                "total_chunks": total_chunks
                            })

                        chunks.append({
                            "id": str(total_chunks),
                            "content": split,
                            "page_number": page_num,
                            "chunk_number_in_page": page_chunk_counters[page_num],
                            "total_chunk_number": total_chunks,
                            "source": uploaded_file.name,
                            "document_name": uploaded_file.name,
                            "source_path": temp_path,
                            "pdf_path": os.path.join(pdf_dir, f"page_{page_num}.pdf"),
                            "image_path": os.path.join(img_dir, f"page_{page_num}.png")
                        })

        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

        with st.spinner(f"Generating embeddings for {len(chunks)} chunks..."):
            # Generate embeddings for all chunks at once
            all_texts = [chunk["content"] for chunk in chunks]
            embeddings = self.batch_embed_texts(all_texts)

            # Add embeddings to chunks
            for chunk, embedding in zip(chunks, embeddings):
                chunk["contentVector"] = embedding

        return chunks, pdf_dir, img_dir

    def regenerate_document_files(self, uploaded_file) -> Tuple[List[dict], str, str]:
        """Regenerate only PDF and image files without vector store processing"""
        doc_name = uploaded_file.name.rsplit('.', 1)[0]
        pdf_dir = os.path.join(self.PDF_PAGES_DIR, doc_name)
        img_dir = os.path.join(self.PDF_IMAGES_DIR, doc_name)

        os.makedirs(pdf_dir, exist_ok=True)
        os.makedirs(img_dir, exist_ok=True)

        # Save the full PDF temporarily
        temp_path = os.path.join(pdf_dir, "temp.pdf")
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getvalue())

        try:
            # Convert PDF to images
            images = pdf2image.convert_from_path(temp_path, dpi=200)
            pdf = PdfReader(temp_path)

            # Save individual pages
            for page_num in range(len(pdf.pages)):
                # Save as PDF
                writer = PdfWriter()
                writer.add_page(pdf.pages[page_num])
                pdf_path = os.path.join(pdf_dir, f"page_{page_num + 1}.pdf")
                with open(pdf_path, "wb") as output_file:
                    writer.write(output_file)

                # Save as image
                img_path = os.path.join(img_dir, f"page_{page_num + 1}.png")
                images[page_num].save(img_path, "PNG")

        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

        return [], pdf_dir, img_dir

    def upload_to_vector_store(self, chunks: List[dict]):
        """Optimized document upload"""
        try:
            # Directly upload all chunks at once instead of batching
            self.vector_store.upload_documents(chunks)
            st.success("Successfully uploaded documents to vector store")
        except Exception as e:
            st.error(f"Error uploading documents: {e}")

    def generate_response(self, query: str, context_chunks: List[dict]) -> str:
        document_context = "\n\n".join([
            f"[Page {chunk['page_number']}]: {chunk['content']}"
            for chunk in context_chunks
        ])

        conversation_context = self.memory.get_conversation_context()

        prompt = f"""Given the following context, conversation history, and question, provide a detailed answer.
        Include page numbers when referring to specific information from the context.

        Document Context:
        {document_context}

        Conversation History:
        {conversation_context}

        Current Question: {query}

        Answer:"""

        messages = [
            {"role": "system", "content": """You are a helpful assistant focused on answering questions about the provided documents. Follow these guidelines:

1. Always cite page numbers when referencing specific information
2. If the answer cannot be found in the provided context, respond with: "I cannot find information about this in the provided documents."
3. If asked questions unrelated to the documents (like personal questions or general knowledge), respond with: "I am designed to help you with questions about the uploaded documents only."
4. Be concise and accurate in your responses
5. Only use information from the provided context - do not make up or infer your own information"""},
            {"role": "user", "content": prompt}
        ]

        return self.llm.generate_completion(messages)

    def chat(self, query: str, k: int = None) -> str:
        if k is None:
            k = CONFIG['top_k_chunks']

        # Add user message to memory
        self.memory.add_message("user", query)

        with st.spinner("Processing your question..."):
            query_vector = self.embeddings.embed_query(query)
            similar_chunks = self.vector_store.search_similar(query_vector, k=k)

            if st.session_state.use_reranking:
                similar_chunks = self.reranker.rerank(query, similar_chunks)


            self.current_similar_chunks = similar_chunks

            if st.session_state.use_image_context:
                # Collect unique image paths
                image_paths = list(set(chunk['image_path'] for chunk in similar_chunks))

                # Create a simpler prompt for image-based context
                prompt = f"""Analyze the provided document pages and answer the following question.
                If you see any page numbers or specific sections, please reference them in your answer.

                Question: {query}"""

                messages = [
                    {"role": "system", "content": "You are a helpful assistant analyzing document pages. Be specific and reference page numbers when possible."},
                    {"role": "user", "content": prompt}
                ]

                response = self.llm.generate_completion_with_images(messages, image_paths)
            else:
                # Original text-based processing
                document_context = "\n\n".join([
                    f"[Page {chunk['page_number']}]: {chunk['content']}"
                    for chunk in similar_chunks
                ])
                conversation_context = self.memory.get_conversation_context()

                # Prepare context more efficiently
                prompt = f"""Answer based on the following context. Include page numbers when referencing specific information.

Context:
{document_context}

Recent conversation:
{conversation_context}

Question: {query}"""

                messages = [
                    {"role": "system", "content": "You are a helpful assistant. Be concise and always cite page numbers when referencing information."},
                    {"role": "user", "content": prompt}
                ]
                response = self.llm.generate_completion(messages)

            # Add response to memory
            self.memory.add_message("assistant", response)

            return response