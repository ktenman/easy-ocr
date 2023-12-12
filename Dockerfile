# Builder image
FROM python:3.9-slim-bookworm as builder

WORKDIR /usr/src/app

COPY . .

# Install packages and clean up in one step to reduce layer size
RUN pip install --no-cache-dir -r requirements.txt \
    && find /usr/local \( -type d -a -name test -o -name tests \) -o \( -type f -a -name '*.pyc' -o -name '*.pyo' \) -exec rm -rf '{}' + \
    && rm -rf /root/.cache

# Final image
FROM python:3.9-slim-bookworm

WORKDIR /usr/src/app

# Copy the Python environment
COPY --from=builder /usr/local /usr/local

# Copy the application code
COPY . .

# Make port 61234 available to the world outside this container
EXPOSE 61234

# Define environment variable
ENV FLASK_APP=app.py

# Run app.py when the container launches
CMD ["flask", "run", "--host=0.0.0.0", "--port=61234"]
