import os
from typing import List, Dict, Tuple
import streamlit as st
from datetime import datetime
import yaml
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
from langchain_openai import AzureOpenAIEmbeddings, OpenAIEmbeddings
from langchain_community.vectorstores.azuresearch import AzureSearch
from PyPDF2 import PdfReader, PdfWriter
import pdf2image
from PIL import Image
import time
from dotenv import load_dotenv

from services.reranker import BM25Reranker
from models.llm_handler import AzureOpenAILLM, OpenAILLM

def load_config():
    """Load configuration from yaml and environment variables"""
    load_dotenv()
    with open('config.yaml', 'r') as file:
        config = yaml.safe_load(file)
    config['openai_api_key'] = os.getenv('OPENAI_API_KEY')
    config['top_k_chunks'] = config.get('top_k_chunks', 5)
    return config

CONFIG = load_config()

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
        self.MAX_FILE_SIZE = 25 * 1024 * 1024  # 25MB
        self.PDF_PAGES_DIR = "pdf_pages"
        self.PDF_IMAGES_DIR = "pdf_images"

        os.makedirs(self.PDF_PAGES_DIR, exist_ok=True)
        os.makedirs(self.PDF_IMAGES_DIR, exist_ok=True)

        # Initialize embeddings based on config
        if CONFIG.get('llm_type') == 'AZURE_OPENAI':
            self.embeddings = AzureOpenAIEmbeddings(
                azure_deployment=CONFIG['azure_deployment'],
                openai_api_version=CONFIG['azure_openai_api_version'],
                azure_endpoint=CONFIG['azure_endpoint'],
                api_key=CONFIG['azure_openai_api_key']
            )
        else:
            self.embeddings = OpenAIEmbeddings(
                model=CONFIG['embedding_model_name'],
                openai_api_key=CONFIG['openai_api_key']
            )

        # Initialize Azure Search vector store
        self.vector_store = AzureSearch(
            azure_search_endpoint=CONFIG['azure_search_endpoint'],
            azure_search_key=CONFIG['azure_search_key'],
            index_name=CONFIG['azure_search_index_name'],
            embedding_function=self.embeddings.embed_query,
            additional_search_client_options={"retry_total": 4}
        )

        if CONFIG['llm_type'] == 'AZURE_OPENAI':
            self.llm = AzureOpenAILLM()
        else:
            self.llm = OpenAILLM()

        self.memory = ConversationMemory()
        self.current_similar_chunks = []
        self.reranker = BM25Reranker()

    def validate_file_size(self, file) -> bool:
        return file.size <= self.MAX_FILE_SIZE

    def process_document(self, uploaded_file) -> Tuple[List[dict], str, str]:
        """Process a document, checking first if it's already been processed"""
        if not self.validate_file_size(uploaded_file):
            raise ValueError(f"File size exceeds maximum limit of 25MB. Current size: {uploaded_file.size / (1024 * 1024):.2f}MB")

        # Check if document was already processed
        processed_docs = self.get_processed_documents()
        if uploaded_file.name in processed_docs:
            doc_name = uploaded_file.name.rsplit('.', 1)[0]
            pdf_dir = os.path.join(self.PDF_PAGES_DIR, doc_name)
            img_dir = os.path.join(self.PDF_IMAGES_DIR, doc_name)

            if os.path.exists(pdf_dir) and os.path.exists(img_dir):
                st.info(f"Document '{uploaded_file.name}' has already been processed and stored.")
                return [], pdf_dir, img_dir
            else:
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
                writer = PdfWriter()
                writer.add_page(pdf.pages[page_num])
                pdf_path = os.path.join(pdf_dir, f"page_{page_num + 1}.pdf")
                with open(pdf_path, "wb") as output_file:
                    writer.write(output_file)

                img_path = os.path.join(img_dir, f"page_{page_num + 1}.png")
                images[page_num].save(img_path, "PNG")

            # Process text and create chunks
            loader = PyPDFLoader(temp_path)
            pages = loader.load()

            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1500,
                chunk_overlap=150,
                separators=["\n\n", "\n", " ", ""]
            )

            chunks = []
            page_chunk_counters = {}
            total_chunks = 0

            with st.spinner("Processing PDF..."):
                for page in pages:
                    splits = text_splitter.split_text(page.page_content)
                    page_num = page.metadata.get("page", 0) + 1

                    if page_num not in page_chunk_counters:
                        page_chunk_counters[page_num] = 0

                    for split in splits:
                        page_chunk_counters[page_num] += 1
                        total_chunks += 1

                        chunks.append({
                            "content": split,
                            "metadata": {
                                "page_number": page_num,
                                "chunk_number_in_page": page_chunk_counters[page_num],
                                "total_chunk_number": total_chunks,
                                "source": uploaded_file.name,
                                "document_name": uploaded_file.name,
                                "pdf_path": os.path.join(pdf_dir, f"page_{page_num}.pdf"),
                                "image_path": os.path.join(img_dir, f"page_{page_num}.png")
                            }
                        })

        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

        # Batch process embeddings
        with st.spinner(f"Generating embeddings for {len(chunks)} chunks..."):
            texts = [chunk["content"] for chunk in chunks]
            embeddings = self.batch_embed_texts(texts)

        return chunks, pdf_dir, img_dir

    def batch_embed_texts(self, texts: List[str], batch_size: int = 32) -> List[List[float]]:
        """Embed multiple texts in batches for better performance"""
        embeddings = []
        total = len(texts)
        progress_bar = st.progress(0)

        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            batch_embeddings = self.embeddings.embed_documents(batch_texts)
            embeddings.extend(batch_embeddings)

            progress = min((i + batch_size) / total, 1.0)
            progress_bar.progress(progress)
            if i % batch_size == 0:
                st.write(f"Processed {i}/{total} embeddings...")

        progress_bar.empty()
        return embeddings

    def upload_to_vector_store(self, chunks: List[dict]):
        """Upload documents to vector store"""
        try:
            texts = [chunk["content"] for chunk in chunks]
            metadatas = [chunk["metadata"] for chunk in chunks]

            self.vector_store.add_texts(
                texts=texts,
                metadatas=metadatas
            )
            st.success("Successfully uploaded documents to vector store")
        except Exception as e:
            st.error(f"Error uploading documents: {e}")

    def get_processed_documents(self) -> set:
        """Get list of already processed document names"""
        try:
            results = self.vector_store.similarity_search(
                query="",  # Empty query to get all documents
                k=1000,  # Large enough to get all documents
                search_type="similarity"
            )
            return {doc.metadata.get('document_name', '') for doc in results}
        except Exception:
            return set()

    def regenerate_document_files(self, uploaded_file) -> Tuple[List[dict], str, str]:
        """Regenerate only PDF and image files without vector store processing"""
        doc_name = uploaded_file.name.rsplit('.', 1)[0]
        pdf_dir = os.path.join(self.PDF_PAGES_DIR, doc_name)
        img_dir = os.path.join(self.PDF_IMAGES_DIR, doc_name)

        os.makedirs(pdf_dir, exist_ok=True)
        os.makedirs(img_dir, exist_ok=True)

        temp_path = os.path.join(pdf_dir, "temp.pdf")
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getvalue())

        try:
            images = pdf2image.convert_from_path(temp_path, dpi=200)
            pdf = PdfReader(temp_path)

            for page_num in range(len(pdf.pages)):
                writer = PdfWriter()
                writer.add_page(pdf.pages[page_num])
                pdf_path = os.path.join(pdf_dir, f"page_{page_num + 1}.pdf")
                with open(pdf_path, "wb") as output_file:
                    writer.write(output_file)

                img_path = os.path.join(img_dir, f"page_{page_num + 1}.png")
                images[page_num].save(img_path, "PNG")

        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

        return [], pdf_dir, img_dir

    def chat(self, query: str, k: int = None) -> str:
        if k is None:
            k = CONFIG['top_k_chunks']

        self.memory.add_message("user", query)

        with st.spinner("Processing your question..."):
            similar_chunks = self.vector_store.similarity_search_with_score(
                query=query,
                k=k
            )

            # Convert to our standard format
            processed_chunks = []
            for doc, score in similar_chunks:
                processed_chunks.append({
                    "content": doc.page_content,
                    "page_number": doc.metadata.get("page_number"),
                    "chunk_number_in_page": doc.metadata.get("chunk_number_in_page"),
                    "total_chunk_number": doc.metadata.get("total_chunk_number"),
                    "image_path": doc.metadata.get("image_path"),
                    "score": score
                })

            if st.session_state.use_reranking:
                processed_chunks = self.reranker.rerank(query, processed_chunks)

            self.current_similar_chunks = processed_chunks

            if st.session_state.use_image_context:
                image_paths = list(set(chunk["image_path"] for chunk in processed_chunks))
                prompt = f"""Analyze the provided document pages and answer the following question.
                If you see any page numbers or specific sections, please reference them in your answer.

                Question: {query}"""

                messages = [
                    {"role": "system", "content": "You are a helpful assistant analyzing document pages. Be specific and reference page numbers when possible."},
                    {"role": "user", "content": prompt}
                ]

                response = self.llm.generate_completion_with_images(messages, image_paths)
            else:
                document_context = "\n\n".join([
                    f"[Page {chunk['page_number']}]: {chunk['content']}"
                    for chunk in processed_chunks
                ])
                conversation_context = self.memory.get_conversation_context()

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

            self.memory.add_message("assistant", response)
            return response