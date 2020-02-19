#!/bin/bash

PYBIND_INCLUDE=`python3 -m pybind11 --includes`
EXTENSION=`python3-config --extension-suffix`

g++ -g -O3 -shared -std=c++14 -fPIC ${PYBIND_INCLUDE} src/grouping.cpp -o src/grouping${EXTENSION}
