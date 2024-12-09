#!/bin/bash
# Download SQLite source code
wget https://www.sqlite.org/2023/sqlite-autoconf-3430100.tar.gz
tar xzf sqlite-autoconf-3430100.tar.gz
cd sqlite-autoconf-3430100

# Configure and build SQLite
./configure --prefix=$HOME/.local
make
make install

# Verify SQLite build
$HOME/.local/bin/sqlite3 --version
