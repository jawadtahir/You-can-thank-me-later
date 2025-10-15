# Use official Python image as base
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Copy code and install dependencies
COPY * ./
RUN pip install --no-cache-dir -r requirements.txt


# Expose Streamlit default port
EXPOSE 8501

VOLUME [ "input_documents" ]

# Pass the OpenAI API key as an environment variable at runtime
# Do not hardcode sensitive data in the Dockerfile

# Allow mounting input_documents at runtime
# (do not copy input_documents, expect it to be mounted)

# Entrypoint for Streamlit
ENTRYPOINT ["streamlit", "run", "frontend.py"]
