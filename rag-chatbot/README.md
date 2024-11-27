# RAG Chatbot

A configurable RAG (Retrieval Augmented Generation) chatbot with support for multiple vector stores and LLM providers.

## Setup

1. Clone the repository
2. Run the setup script:
   ```bash
   python start.py
   ```
3. Edit the .env file with your credentials
4. Start the application:
   ```bash
   python start.py --start
   ```

## Features

- Support for both Azure OpenAI and OpenAI
- Vector store options: Azure Cognitive Search and ChromaDB
- Conversation memory
- PDF document processing
- Streamlit web interface

## Project Structure

```
rag-chatbot/
├── src/
│   ├── app.py                 # Main Streamlit application
│   ├── models/                # Model definitions
│   ├── services/             # Service implementations
│   ├── utils/                # Utility functions
│   └── config/               # Configuration files
├── tests/                    # Test files
├── data/                     # Data directory
│   ├── raw/                  # Raw data
│   └── processed/            # Processed data
├── logs/                     # Log files
├── requirements.txt          # Project dependencies
└── README.md                # Project documentation
```