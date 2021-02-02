#!/bin/bash -e
python3 setup.py clean sdist

LATEST_RELEASE="dist/$(ls -t1 dist|  head -n 1)"

pip install $LATEST_RELEASE

echo *************************Success**************************
echo  ":" $LATEST_RELEASE
echo "**********************************************************"