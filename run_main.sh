#!/bin/bash
conda init bash
conda activate py310
until python main.py
do
    echo "restarting"
    sleep 3
done
