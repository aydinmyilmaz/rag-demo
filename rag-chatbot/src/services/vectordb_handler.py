import os
from typing import List, Dict, Tuple
from langchain_community.vectorstores.azuresearch import AzureSearch
from langchain_openai import AzureOpenAIEmbeddings, OpenAIEmbeddings
import streamlit as st
import yaml
from dotenv import load_dotenv

def load_config():
    """Load configuration from yaml and environment variables"""
    load_dotenv()

    with open('config.yaml', 'r') as file:
        config = yaml.safe_load(file)

    config['openai_api_key'] = os.getenv('OPENAI_API_KEY')
    config['top_k_chunks'] = config.get('top_k_chunks', 5)

    return config

CONFIG = load_config()

class VectorStore:
    def __init__(self):
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
                api_key=CONFIG['openai_api_key']
            )

        # Initialize Azure AI Search vector store
        self.vector_store = AzureSearch(
            azure_search_endpoint=CONFIG['azure_search_endpoint'],
            azure_search_key=CONFIG['azure_search_password'],
            index_name=CONFIG['azure_search_index_name'],
            embedding_function=self.embeddings.embed_query,
            # Configure retries for Azure client
            additional_search_client_options={"retry_total": 4}
        )

    def get_processed_documents(self) -> set:
        """Get list of already processed document names"""
        try:
            # Use the metadatas to get document names
            results = self.vector_store.similarity_search(
                "dummy query",  # We just need to get metadata
                k=1000,  # Large enough to get all documents
                include_metadata=True
            )
            return {doc.metadata.get('document_name', '') for doc in results}
        except Exception:
            return set()

    def search_similar(self, query: str, k: int) -> List[dict]:
        """Search for similar documents using vector similarity"""
        try:
            results = self.vector_store.similarity_search_with_score(
                query=query,
                k=k
            )

            similar_chunks = []
            for doc, score in results:
                metadata = doc.metadata

                # Get metadata values with explicit type conversion
                source = metadata.get("source", "Unknown")
                page_num = metadata.get("page_number", "Unknown")
                chunk_in_page = int(metadata.get("chunk_number_in_page", 0))
                total_chunk = int(metadata.get("total_chunk_number", 0))

                # Construct the document name without extension
                doc_name = source.rsplit('.', 1)[0]

                # Construct the paths
                pdf_path = os.path.join("pdf_pages", doc_name, f"page_{page_num}.pdf")
                img_path = os.path.join("pdf_images", doc_name, f"page_{page_num}.png")

                similar_chunks.append({
                    "content": doc.page_content,
                    "page_number": page_num,
                    "chunk_number_in_page": chunk_in_page,
                    "total_chunk_number": total_chunk,
                    "source": source,
                    "pdf_path": pdf_path,
                    "image_path": img_path,
                    "source_path": metadata.get("source_path", ""),
                    "score": score
                })

            return similar_chunks

        except Exception as e:
            st.error(f"Error in search_similar: {str(e)}")
            raise

    def upload_documents(self, documents: List[dict]):
        """Upload documents to Azure AI Search"""
        try:
            # Convert documents to LangChain Document format
            texts = [doc["content"] for doc in documents]
            metadatas = [{
                "page_number": doc["page_number"],
                "source": doc["source"],
                "source_path": doc.get("source_path", ""),
                "document_name": doc["document_name"],
                "chunk_number_in_page": doc["chunk_number_in_page"],
                "total_chunk_number": doc["total_chunk_number"]
            } for doc in documents]

            # Add documents in batches
            batch_size = 100
            for i in range(0, len(documents), batch_size):
                batch_end = min(i + batch_size, len(documents))
                self.vector_store.add_texts(
                    texts=texts[i:batch_end],
                    metadatas=metadatas[i:batch_end]
                )

            st.success("Successfully uploaded documents to Azure AI Search")

        except Exception as e:
            st.error(f"Error in upload_documents: {str(e)}")
            raise