#!/bin/bash
# Demonstration script for DeepPath
# Assumes dependencies and dataset are already installed via .codex/setup.sh

set -e

RELATION=${1:-athletePlaysForTeam}

# Run training and testing for the provided relation
./pathfinder.sh "$RELATION"
