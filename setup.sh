#!/bin/bash

# Install a specific version of SQLite
wget https://www.sqlite.org/2024/sqlite-autoconf-3420000.tar.gz
tar -xvf sqlite-autoconf-3420000.tar.gz
cd sqlite-autoconf-3420000
./configure
make
make install

# Verify installation
sqlite3 --version
