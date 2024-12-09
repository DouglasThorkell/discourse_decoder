#!/bin/bash

# Install a specific version of SQLite
wget https://www.sqlite.org/2023/sqlite-autoconf-3430100.tar.gz
tar -xvf sqlite-autoconf-3430100.tar.gz
cd sqlite-autoconf-3430100

# Configure, build, and install SQLite
./configure --prefix=$HOME/.local
make
make install

# Update PATH to use the newly installed SQLite
export PATH=$HOME/.local/bin:$PATH

# Verify installation
sqlite3 --version
