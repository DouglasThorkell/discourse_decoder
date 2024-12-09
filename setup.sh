#!/bin/bash

# Set up SQLite build directory
SQLITE_VERSION=3430100
SQLITE_NAME="sqlite-autoconf-${SQLITE_VERSION}"
INSTALL_DIR="$HOME/.local"

# Download and extract SQLite
wget "https://www.sqlite.org/2023/${SQLITE_NAME}.tar.gz"
tar -xvf "${SQLITE_NAME}.tar.gz"
cd "${SQLITE_NAME}"

# Build and install SQLite locally
./configure --prefix="${INSTALL_DIR}"
make
make install

# Update PATH to prioritize locally installed SQLite
export PATH="${INSTALL_DIR}/bin:$PATH"

# Verify the installed version
sqlite3 --version
