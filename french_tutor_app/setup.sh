#!/bin/bash

# Clone the GitHub repository
git clone https://https://github.com/nathanvdv/Nvidia/french_tutor_app

# Navigate to the project directory
cd french_tutor_app

# Create a virtual environment (optional but recommended)
python -m venv venv

# Activate the virtual environment
source venv/bin/activate

# Install the required packages from the requirements.txt file
pip install -r requirements.txt