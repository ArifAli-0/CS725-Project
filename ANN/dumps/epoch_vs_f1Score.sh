#!/bin/bash

echo "Epoch,F1_SCORE" >log_f1score.txt
# Extract and format F1 scores for each unique epoch in CSV format
awk '/Epoch/ {epoch = $2} /F1 Score/ {gsub(/%/, ""); f1_score = $NF; printf "%s%s\n", epoch, f1_score}' log_training.txt >> log_f1score.txt

python3 epoch_vs_f1Score.py

#rm log_f1score.txt