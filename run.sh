#!/bin/bash

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}Setting up neural network environment...${NC}"

# Create virtual environment if it doesn't exist
if [ ! -d ".venv" ]; then
    echo -e "${GREEN}Creating virtual environment...${NC}"
    python3 -m venv .venv
fi

# Activate virtual environment
source .venv/bin/activate

# Install requirements
echo -e "${GREEN}Installing dependencies...${NC}"
pip install -r requirements.txt

# Run the neural network
echo -e "${GREEN}Starting neural network...${NC}"
python nn.py

# Deactivate virtual environment
deactivate 