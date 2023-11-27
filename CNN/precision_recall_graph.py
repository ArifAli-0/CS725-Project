#!/usr/bin/python3

import pandas as pd
import matplotlib.pyplot as plt

# Read the CSV file into a DataFrame
data = pd.read_csv('precision_recall.txt')

# Plot precision and recall
plt.plot(data['Epoch'], data['Precision'], marker='o', linestyle='-', label='Precision', color='red')
plt.plot(data['Epoch'], data['Recall'], marker='*', linestyle='-', label='Recall', color='blue')
plt.title('Epoch vs Precision and Recall')
plt.xlabel('Epoch')
plt.ylabel('Score')
plt.xticks(data['Epoch'])
plt.grid(True)
plt.legend(loc=0)
plt.savefig("precision_recall_graph.png")
#plt.show()