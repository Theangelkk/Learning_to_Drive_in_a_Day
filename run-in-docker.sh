#!/bin/bash

# Usage: ./run-in-docker.sh
#
# NOTE: Assumed that laptop has Intel Graphics. Remove "/dev/dri" mount to use virtual grpahics (slow).

docker run --name prova1 --network="host" \
    --rm \
    -ti \
    -e DISPLAY \
    -e DONKEY_SIM_HEADLESS=0 \
    -e DONKEY_SIM_PATH=/Users/angelocasolaropro/sim/donkey_sim.app/Contents/MacOS/donkey_sim \
    -v $HOME/.Xauthority:/root/.Xauthority \
    -v $HOME/sim:/sim \
    -v $(pwd):/code \
    -w /code \
    9ad9924abe24 ./run.py
