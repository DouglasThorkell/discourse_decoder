# Use a base Python image with Linux
FROM python:3.9-slim

# Install system dependencies (SQLite and its development library)
RUN apt-get update && apt-get install -y sqlite3 libsqlite3-dev

# Set the working directory
WORKDIR /app

# Copy all files from your repository into the container
COPY . /app

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose the port Streamlit uses
EXPOSE 8501

# Command to run your Streamlit app
CMD ["streamlit", "run", "discoursedecoder.py"]
