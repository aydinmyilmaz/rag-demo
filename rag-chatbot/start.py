import os
import sys
import subprocess
import platform
import venv
from pathlib import Path
from datetime import datetime
import shutil

class Colors:
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    END = '\033[0m'

    @staticmethod
    def supports_color() -> bool:
        plat = platform.system().lower()
        supported_platform = plat != 'windows' or 'ANSICON' in os.environ
        is_a_tty = hasattr(sys.stdout, 'isatty') and sys.stdout.isatty()
        return supported_platform and is_a_tty

    @classmethod
    def blue(cls, text: str) -> str:
        return f"{cls.BLUE}{text}{cls.END}" if cls.supports_color() else text

    @classmethod
    def green(cls, text: str) -> str:
        return f"{cls.GREEN}{text}{cls.END}" if cls.supports_color() else text

    @classmethod
    def yellow(cls, text: str) -> str:
        return f"{cls.YELLOW}{text}{cls.END}" if cls.supports_color() else text

    @classmethod
    def red(cls, text: str) -> str:
        return f"{cls.RED}{text}{cls.END}" if cls.supports_color() else text

def log(message: str, level: str = 'info') -> None:
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    color_map = {
        'info': Colors.blue,
        'success': Colors.green,
        'warning': Colors.yellow,
        'error': Colors.red
    }
    color_func = color_map.get(level, Colors.blue)
    print(f"[{color_func(timestamp)}] {message}")

def create_file(path: Path, content: str = "") -> None:
    """Create a file with the given content"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)
    log(f"Created file: {path}", 'success')

def create_project_structure():
    """Create the project structure with all necessary files"""
    root = Path.cwd()

    # Create main directories
    directories = [
        "src",
        "src/config",
        "src/models",
        "src/services",
        "src/utils",
        "tests",
        "data/raw",
        "data/processed",
        "logs",
        "chroma_db"
    ]

    for dir_path in directories:
        path = root / dir_path
        path.mkdir(parents=True, exist_ok=True)
        log(f"Created directory: {dir_path}", 'success')

    # Create configuration files
    config_files = {
        ".env.example": """OPENAI_API_KEY=your_openai_api_key
AZURE_SEARCH_ENDPOINT=your_azure_search_endpoint
AZURE_SEARCH_KEY=your_azure_search_key
AZURE_SEARCH_INDEX_NAME=your_index_name
AZURE_OPENAI_ENDPOINT=your_azure_openai_endpoint
AZURE_OPENAI_KEY=your_azure_openai_key
AZURE_DEPLOYMENT_NAME=your_deployment_name""",

        "requirements.txt": """# Core requirements
streamlit
openai
langchain
python-dotenv

# Vector stores
chromadb
azure-search-documents

# Data processing
numpy
pandas
PyPDF2
pypdf

# Azure
azure-core

# Additional dependencies
pyyaml
tqdm
requests""",

        ".gitignore": """# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
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
```"""
    }

    for filename, content in config_files.items():
        create_file(root / filename, content)

    # Create python files
    python_files = [
        "src/__init__.py",
        "src/models/__init__.py",
        "src/services/__init__.py",
        "src/utils/__init__.py",
        "src/config/__init__.py",
        "tests/__init__.py"
    ]

    for filepath in python_files:
        create_file(root / filepath, "")

class ProjectSetup:
    def __init__(self):
        self.project_root = Path.cwd()
        self.venv_path = self.project_root / 'venv'
        self.is_windows = platform.system().lower() == 'windows'
        self.python_cmd = 'python' if self.is_windows else 'python3'
        self.pip_cmd = str(self.venv_path / 'Scripts' / 'pip.exe') if self.is_windows else str(self.venv_path / 'bin' / 'pip')
        self.python_venv = str(self.venv_path / 'Scripts' / 'python.exe') if self.is_windows else str(self.venv_path / 'bin' / 'python')

    def setup(self):
        """Run the complete setup process"""
        log("Starting project setup...")

        # Create project structure
        create_project_structure()

        # Create virtual environment
        log("Creating virtual environment...")
        venv.create(self.venv_path, with_pip=True)

        # Install requirements
        log("Installing requirements...")
        subprocess.run([self.pip_cmd, "install", "--upgrade", "pip"])
        subprocess.run([self.pip_cmd, "install", "-r", "requirements.txt"])

        log("Project setup completed successfully!", 'success')

def main():
    setup = ProjectSetup()

    if len(sys.argv) > 1 and sys.argv[1] == "--clean":
        # Clean up everything except start.py
        for path in Path.cwd().iterdir():
            if path.name != "start.py":
                if path.is_dir():
                    shutil.rmtree(path)
                else:
                    path.unlink()
        log("Cleaned up project directory.", 'success')

    setup.setup()

if __name__ == "__main__":
    main()