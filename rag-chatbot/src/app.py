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
from services.rag_chatbot import RAGChatbot
from services.vectordb_handler import ChromaStore, AzureSearchStore
from models.llm_handler import AzureOpenAILLM, OpenAILLM
# Load configuration
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

def reset_conversation():
    st.session_state.messages = []
    st.session_state.chatbot.memory.messages = []
    st.rerun()


import streamlit as st
from typing import List, Dict
import os

def initialize_session_state():
    """Initialize all session state variables"""
    if 'chatbot' not in st.session_state:
        st.session_state.chatbot = RAGChatbot()
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'upload_visible' not in st.session_state:
        st.session_state.upload_visible = False
    if 'context_visible' not in st.session_state:
        st.session_state.context_visible = False
    if 'settings_visible' not in st.session_state:
        st.session_state.settings_visible = False
    if 'k_value' not in st.session_state:
        st.session_state.k_value = 5
    if 'use_image_context' not in st.session_state:
        st.session_state.use_image_context = False
    if 'use_reranking' not in st.session_state:
        st.session_state.use_reranking = False

def show_chat_interface():
    """Separate function for chat interface"""
    chat_container = st.container()

    with chat_container:
        # Show welcome message if no messages

        # Display messages
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.write(msg["content"])

        # Add space at bottom to prevent content being hidden behind chat input
        st.markdown("<div style='height: 100px'></div>", unsafe_allow_html=True)

    # Chat input
    if prompt := st.chat_input("💭 Ask about your documents...", key="chat_input"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        response = st.session_state.chatbot.chat(
            prompt,
            k=st.session_state.get("k_value", 5)
        )
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.rerun()

def show_active_panel():
    """Separate function for panel content"""
    if st.session_state.settings_visible:
        st.markdown("### 🛠️ Settings")
        st.slider("Number of similar chunks (k)", 1, 10, 5, key="k_value")
        col1, col2 = st.columns(2)
        with col1:
            st.toggle("Use Image Context", value=False, key="use_image_context",
                     help="Show source document images alongside responses")
        with col2:
            st.toggle("Use BM25 Reranking", value=False, key="use_reranking",
                     help="Improve response relevance using BM25 algorithm")

    elif st.session_state.upload_visible:

        # File uploader without expander
        uploaded_file = st.file_uploader(
            "📥 Upload a PDF document",
            type="pdf",
            help="Maximum file size: 25MB"
        )

        if uploaded_file:
            file_size = uploaded_file.size / (1024 * 1024)
            col1, col2 = st.columns([3, 1])
            with col1:
                st.markdown(f"**Selected file:** {uploaded_file.name} ({file_size:.1f}MB)")
            with col2:
                process_button = st.button(
                    "✨ Process File",
                    type="primary",
                    use_container_width=True
                )

            if process_button:
                # Process document in a status container
                with st.status("Processing document...", expanded=True) as status:
                    try:
                        chunks, pdf_dir, img_dir = st.session_state.chatbot.process_document(uploaded_file)
                        st.session_state.chatbot.upload_to_vector_store(chunks)
                        status.update(label="✅ Document processed successfully!", state="complete")
                    except Exception as e:
                        st.error(f"Error processing document: {str(e)}")

        # Show processed documents list
        st.divider()
        if isinstance(st.session_state.chatbot.vector_store, ChromaStore):
            docs = st.session_state.chatbot.vector_store.get_processed_documents()
            if docs:
                st.markdown("### 📑 Processed Documents")
                for doc in sorted(docs):
                    st.markdown(f"📄 {doc}")
                if hasattr(st.session_state.chatbot.vector_store, 'collection'):
                    try:
                        doc_count = len(st.session_state.chatbot.vector_store.collection.get()['ids'])
                        st.caption(f"📊 Total chunks: {doc_count}")
                    except:
                        pass
            else:
                st.info("👋 Upload your first document to get started!")

    elif st.session_state.context_visible:
        st.markdown("### 🎯 Source Context")
        if (hasattr(st.session_state.chatbot, 'current_similar_chunks') and
            st.session_state.chatbot.current_similar_chunks):

            # Display chunks in tabs
            tabs = st.tabs([f"Chunk {i+1} (Page {chunk['page_number']})"
                          for i, chunk in enumerate(st.session_state.chatbot.current_similar_chunks)])

            for i, (tab, chunk) in enumerate(zip(tabs, st.session_state.chatbot.current_similar_chunks)):
                with tab:
                    col1, col2 = st.columns([1, 1])
                    with col1:
                        st.markdown("**📝 Extracted Text:**")
                        st.markdown(f"```\n{chunk['content']}\n```")

                    with col2:
                        st.markdown("**🖼️ Original Page:**")
                        image_path = chunk.get('image_path', '')
                        if image_path and os.path.exists(image_path):
                            try:
                                image = Image.open(image_path)
                                st.image(image, caption=f"Page {chunk['page_number']}")
                            except Exception as e:
                                st.error(f"Error loading image: {str(e)}")
                        else:
                            st.info("No image available for this chunk")

                    st.caption(f"Page {chunk['page_number']} | Chunk {chunk['chunk_number_in_page']}/{chunk['total_chunk_number']}")
        else:
            st.info("💫 No context chunks available yet. Start a conversation to see relevant content.")

def main():
    initialize_session_state()

    # Enhanced CSS for responsive layout
    # Add this updated CSS section in your main function:
    st.markdown("""
        <style>
        /* Base styles */
        .main .block-container {
            padding-top: 2rem;
            padding-bottom: 5rem;  /* Increased padding for chat input */
            transition: all 0.3s ease;
        }

        /* Sidebar styles */
        section[data-testid="stSidebar"] {
            width: 300px !important;
            background-color: #f8f9fa;
            text-align: center;
        }

        /* Button styles */
        .stButton button {
            width: 100%;
        }

        /* Chat input container styles */
        .stChatFloatingInputContainer {
            bottom: 20px !important;
            background-color: white;
            padding: 1rem !important;
            border-radius: 10px;
            box-shadow: 0 -5px 15px rgba(0,0,0,0.1);
            max-width: none !important;
        }

        .stChatInputContainer {
            padding: 10px !important;
        }

        .stChatInputContainer textarea {
            font-size: 1.1rem !important;
            padding: 15px !important;
            min-height: 50px !important;
        }

        /* Responsive layout */
        @media (min-width: 1200px) {
            .main .block-container {
                max-width: calc(100% - 600px) !important;
                margin-left: 300px !important;
                padding-left: 2rem !important;
                padding-right: 2rem !important;
            }

            .panel-container {
                position: fixed;
                left: 300px;
                top: 0;
                width: 300px;
                height: 100vh;
                overflow-y: auto;
                padding: 2rem 1rem;
                background-color: white;
                border-right: 1px solid #ddd;
            }

            /* Chat container positioning */
            .chat-container {
                margin-left: 300px;
                padding: 0 2rem;
                margin-bottom: 100px;  /* Space for chat input */
            }

            .stChatFloatingInputContainer {
                right: 20px !important;
                left: 320px !important;
                width: calc(100% - 640px) !important;
            }
        }

        @media (max-width: 1199px) {
            .panel-container {
                margin-bottom: 2rem;
            }

            .chat-container {
                margin-top: 2rem;
                margin-bottom: 100px;  /* Space for chat input */
            }

            .stChatFloatingInputContainer {
                left: 20px !important;
                right: 20px !important;
                width: calc(100% - 40px) !important;
            }
        }

        /* Active button styles */
        .stButton button[data-testid="baseButton-primary"] {
            background-color: #ff4b4b !important;
            border-color: #ff4b4b !important;
            color: white !important;
        }

        /* Message container spacing */
        .stChatMessage {
            margin-bottom: 1rem !important;
            padding: 1rem !important;
        }

        /* Add spacing before chat input */
        .chat-container {
            min-height: calc(100vh - 200px);  /* Ensure content doesn't get hidden behind chat input */
        }

        /* Transitions */
        .panel-container, .chat-container {
            transition: all 0.3s ease;
        }
        </style>
    """, unsafe_allow_html=True)

    # Sidebar
    with st.sidebar:
        st.markdown("<h1 style='text-align: center;'>🧭 Navigation</h1>", unsafe_allow_html=True)

        # Navigation buttons
        button_type = "primary" if st.session_state.upload_visible else "secondary"
        if st.button("📥 Upload Files", type=button_type, use_container_width=True):
            st.session_state.upload_visible = not st.session_state.upload_visible
            st.session_state.context_visible = False
            st.session_state.settings_visible = False
            st.rerun()

        button_type = "primary" if st.session_state.context_visible else "secondary"
        if st.button("🎯 Source Context", type=button_type, use_container_width=True):
            st.session_state.context_visible = not st.session_state.context_visible
            st.session_state.upload_visible = False
            st.session_state.settings_visible = False
            st.rerun()

        button_type = "primary" if st.session_state.settings_visible else "secondary"
        if st.button("🛠️ Settings", type=button_type, use_container_width=True):
            st.session_state.settings_visible = not st.session_state.settings_visible
            st.session_state.upload_visible = False
            st.session_state.context_visible = False
            st.rerun()

        st.divider()

        # Start New Chat button
        if st.button("✨ Start New Chat", type="primary", use_container_width=True):
            st.session_state.messages = []
            st.session_state.chatbot.memory.messages = []
            st.session_state.chatbot.current_similar_chunks = []
            st.session_state.context_visible = False
            st.session_state.upload_visible = False
            st.session_state.settings_visible = False
            st.rerun()

        st.divider()

        # Configuration display
        st.markdown("<div style='text-align: center;'><h4>⚙️ Configuration</h4></div>", unsafe_allow_html=True)
        st.markdown(f"<div style='text-align: center;'>📊 Vector Store: {CONFIG['vector_store']}</div>", unsafe_allow_html=True)
        st.markdown(f"<div style='text-align: center;'>🤖 LLM Type: {CONFIG['llm_type']}</div>", unsafe_allow_html=True)
        st.markdown(f"<div style='text-align: center;'>✨ Chunks (k): {st.session_state.k_value}</div>", unsafe_allow_html=True)

    # Main content area
    st.title("📚 Document Chatbot 🤖")
    if len(st.session_state.messages) == 0:
            st.markdown("### 👋 Welcome! Upload a document and start chatting!")
    st.divider()

    # Create containers for panels and chat
    if any([st.session_state.upload_visible,
            st.session_state.context_visible,
            st.session_state.settings_visible]):
        panel_container = st.container()
        with panel_container:
            show_active_panel()

    # Chat container
    chat_container = st.container()
    with chat_container:
        show_chat_interface()

if __name__ == "__main__":
    main()