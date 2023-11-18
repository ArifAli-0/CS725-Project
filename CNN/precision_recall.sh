#!/bin/bash

# Extract and format precision and recall for each unique epoch in CSV format
echo "Epoch,Precision,Recall" > precision_recall.txt
awk '/Epoch/ {epoch = $2} /Precision:/ {precision = $2} /Recall:/ {recall = $2; print epoch precision "," recall}' training_log.txt >> precision_recall.txt

python3 precision_recall_graph.py

rm precision_recall.txt
