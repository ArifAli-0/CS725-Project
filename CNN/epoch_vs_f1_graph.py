#!/usr/bin/python3

import pandas as pd
import matplotlib.pyplot as plt

# Read the CSV file into a DataFrame
df = pd.read_csv('f1.txt')

df['Epoch']=df['Epoch'].astype(int)

# Plot the epoch vs F1_score
plt.plot(df['Epoch'], df['F1_SCORE'], marker='o', linestyle='-')
plt.title('Epoch vs F1 Score')
plt.xlabel('Epoch')
plt.ylabel('F1 Score')
plt.grid(True)
plt.xticks(df['Epoch'])
plt.savefig("epoch_vs_f1score.png")
#plt.show()


