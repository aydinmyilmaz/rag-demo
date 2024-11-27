import os
import sys

def create_directory(path):
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"Created directory: {path}")

def create_file(path, content=""):
    with open(path, 'w') as f:
        f.write(content)
    print(f"Created file: {path}")

def create_project_structure():
    # Project root directory
    root_dir = "rag-chatbot"
    create_directory(root_dir)

    # Create main project directories
    dirs = [
        "src",
        "src/utils",
        "src/models",
        "src/services",
        "src/config",
        "tests",
        "data",
        "data/raw",
        "data/processed",
        "logs",
        "notebooks"
    ]

    for dir_path in dirs:
        create_directory(os.path.join(root_dir, dir_path))

    # Create files
    files = {
        "requirements.txt": "",
        ".env.example": """OPENAI_API_KEY=your_openai_api_key
AZURE_SEARCH_ENDPOINT=your_azure_search_endpoint
AZURE_SEARCH_KEY=your_azure_search_key
AZURE_SEARCH_INDEX_NAME=your_index_name
AZURE_OPENAI_ENDPOINT=your_azure_openai_endpoint
AZURE_OPENAI_KEY=your_azure_openai_key
AZURE_DEPLOYMENT_NAME=your_deployment_name""",
        ".gitignore": """# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Virtual Environment
venv/
ENV/

# IDE
.idea/
.vscode/
*.swp
*.swo

# Project specific
.env
logs/
data/raw/*
data/processed/*
!data/raw/.gitkeep
!data/processed/.gitkeep
chroma_db/
temp.pdf""",
        "README.md": """# RAG Chatbot

A configurable RAG (Retrieval Augmented Generation) chatbot with support for multiple vector stores and LLM providers.

## Setup

1. Clone the repository
2. Create a virtual environment: `python -m venv venv`
3. Activate the virtual environment:
   - Windows: `venv\\Scripts\\activate`
   - Unix/MacOS: `source venv/bin/activate`
4. Install dependencies: `pip install -r requirements.txt`
5. Copy `.env.example` to `.env` and fill in your credentials
6. Run the application: `streamlit run src/app.py`

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
│   ├── services/              # Service implementations
│   ├── utils/                 # Utility functions
│   └── config/               # Configuration files
├── tests/                    # Test files
├── data/                     # Data directory
│   ├── raw/                  # Raw data
│   └── processed/            # Processed data
├── logs/                     # Log files
├── notebooks/               # Jupyter notebooks
├── requirements.txt         # Project dependencies
└── README.md               # Project documentation
```""",
        "src/app.py": "",
        "src/config/config.py": "",
        "src/models/__init__.py": "",
        "src/models/memory.py": "",
        "src/models/vector_store.py": "",
        "src/models/llm.py": "",
        "src/services/__init__.py": "",
        "src/services/document_processor.py": "",
        "src/services/chat_service.py": "",
        "src/utils/__init__.py": "",
        "src/utils/logger.py": "",
        "tests/__init__.py": "",
        "tests/test_document_processor.py": "",
        "tests/test_chat_service.py": "",
        "tests/test_vector_store.py": "",
        "data/raw/.gitkeep": "",
        "data/processed/.gitkeep": "",
        "notebooks/examples.ipynb": "",
    }

    for file_path, content in files.items():
        create_file(os.path.join(root_dir, file_path), content)

if __name__ == "__main__":
    print("Creating RAG Chatbot project structure...")
    create_project_structure()
    print("\nProject structure created successfully!")
    print("""
Next steps:
1. cd rag-chatbot
2. python -m venv venv
3. Activate your virtual environment:
   - Windows: .\\venv\\Scripts\\activate
   - Unix/MacOS: source venv/bin/activate
4. pip install -r requirements.txt
5. Copy .env.example to .env and fill in your credentials
6. Start developing!
    """)