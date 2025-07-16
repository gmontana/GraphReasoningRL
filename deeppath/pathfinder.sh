#!/bin/bash

# Check if a relation was provided
if [ -z "$1" ]; then
  echo "Usage: $0 <relation_name>"
  echo "Example: $0 athletePlaysForTeam"
  exit 1
fi

relation=$1
mode=${2:-"train_test"}  # Default mode is train_test

# Run the main script with the provided relation
python main.py $relation --mode $mode