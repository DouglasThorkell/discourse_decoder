# Use a lightweight base Python image
FROM python:3.9-slim

# Install build tools and dependencies
RUN apt-get update && apt-get install -y \
    wget \
    build-essential \
    libreadline-dev \
    zlib1g-dev \
    libsqlite3-dev

# Install SQLite from source (latest version)
RUN wget https://www.sqlite.org/2024/sqlite-autoconf-3420000.tar.gz \
    && tar -xzf sqlite-autoconf-3420000.tar.gz \
    && cd sqlite-autoconf-3420000 \
    && ./configure \
    && make \
    && make install \
    && cd .. \
    && rm -rf sqlite-autoconf-3420000*

# Verify SQLite installation
RUN sqlite3 --version

# Set the working directory
WORKDIR /app

# Copy all files from the repository into the container
COPY . /app

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose Streamlit's default port
EXPOSE 8501

# Command to run the Streamlit app
CMD ["streamlit", "run", "discoursedecoder.py"]

