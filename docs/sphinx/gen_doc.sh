#!/bin/bash
rm -r build/html
rm -r source/*.rst
sphinx-apidoc -M -e -o source/ ../../src
mv source/modules.rst source/index.rst
make html
mv build/html/src.html build/html/index.html

