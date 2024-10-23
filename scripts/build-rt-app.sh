#!/usr/bin/env bash

SCRIPT_PATH=$(dirname "$0")
ORIG_PATH=$(pwd)

cd ${SCRIPT_PATH}/..
git submodule init
git submodule update
cd rt-app

git clone https://github.com/json-c/json-c.git
cd json-c
export ac_cv_func_malloc_0_nonnull=yes
export ac_cv_func_realloc_0_nonnull=yes
cmake .
make
cd ..
export LIBRARY_PATH=$(pwd)/json-c/
export C_INCLUDE_PATH=$(pwd)
./autogen.sh
./configure --disable-shared --enable-static
make
cd ${ORIG_PATH}

