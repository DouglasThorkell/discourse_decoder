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

# Clone pysqlite3 repository
git clone https://github.com/coleifer/pysqlite3.git
cd pysqlite3

# Copy SQLite amalgamation files into the pysqlite3 directory
cp ../sqlite-autoconf-3430100/sqlite3.[ch] .

# Build pysqlite3 using the custom SQLite files
python setup.py build_static build
python setup.py install --user
