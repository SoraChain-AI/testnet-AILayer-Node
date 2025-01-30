#!/bin/bash

# Check if at least one argument is passed
if [ $# -eq 0 ]; then
    echo "No arguments provided. Please provide arguments."
    exit 1
fi

# The first argument is the script to run and the rest are passed to the Python script
python3 submitJob.py "$@"
