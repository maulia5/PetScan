#!/bin/bash

# Check if the script is run as root
# if [ "$EUID" -ne 0 ]; then
#     echo "Please run this script as root"
#     exit 1
# fi

# Install dependencies
# apt-get update
# apt-get install -y python3-pip python3-opencv python3-tk

# Install requirements
# pip3 install -r requirements.txt

# Run training and testing
python3 build_centroid.py
python3 build_embedding.py
#python3 test_and_eval.py
