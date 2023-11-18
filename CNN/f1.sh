#!/bin/bash

echo "Epoch,F1_SCORE" >f1.txt
# Extract and format F1 scores for each unique epoch in CSV format
awk '/Epoch/ {epoch = $2} /F1 Score/ {gsub(/%/, ""); f1_score = $NF; printf "%s%s\n", epoch, f1_score}' training_log.txt >> f1.txt

python3 epoch_vs_f1_graph.py


