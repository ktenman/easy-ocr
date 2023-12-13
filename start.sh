#!/bin/bash

# Activate the virtual environment
source /Users/tenman/easy-ocr/venv/bin/activate

# Run the Python script
python3 /Users/tenman/easy-ocr/app-redis.py > /Users/tenman/easy-ocr/logs/app-redis.log 2>&1 &
