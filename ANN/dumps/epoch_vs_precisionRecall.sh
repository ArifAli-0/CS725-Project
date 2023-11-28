#!/bin/bash

# Extract and format precision and recall for each unique epoch in CSV format
echo "Epoch,Precision,Recall" > log_precisionRecall.txt
awk '/Epoch/ {epoch = $2} /Precision:/ {precision = $2} /Recall:/ {recall = $2; print epoch precision "," recall}' log_training.txt >> log_precisionRecall.txt

python3 epoch_vs_precisionRecall.py

#rm log_precisionRecall.txt
