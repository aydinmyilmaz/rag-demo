#!/bin/bash

# Check if Python virtual environment exists
if [ ! -d "venv" ]; then
    echo "Creating Python virtual environment..."
    python3.11 -m venv venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Check and install required packages
echo "Installing/Updating required packages..."
pip install -r requirements.txt

# Check if the app.py exists in src directory
if [ ! -f "src/app.py" ]; then
    echo "Error: src/app.py not found!"
    exit 1
fi

# Run Streamlit application
echo "Starting Streamlit application..."
streamlit run src/app.py --server.port=8000 --server.address=0.0.0.0

# Deactivate virtual environment on script exit
trap 'deactivate' EXIT