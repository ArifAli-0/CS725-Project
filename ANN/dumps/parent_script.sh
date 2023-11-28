#!/bin/bash

python3 ann.py > log_training.txt

bash epoch_vs_f1Score.sh
bash epoch_vs_precisionRecall.sh
