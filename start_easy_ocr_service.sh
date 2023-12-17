#!/bin/bash

ENV_LOG_FILE="/Users/tenman/easy-ocr/env.log"

export RABBITMQ_USER=
export RABBITMQ_PASSWORD=
export REDIS_PASSWORD=

# Log environment variables for debugging
env > "$ENV_LOG_FILE"

# Activate the virtual environment
source /Users/tenman/easy-ocr/venv/bin/activate

# Run the Python script and redirect output to log file
# python3 /Users/tenman/easy-ocr/app-redis.py
python3 /Users/tenman/easy-ocr/app-rabbitmq.py
