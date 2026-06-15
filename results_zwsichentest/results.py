import matplotlib.pyplot as plt
import numpy as np 
import csv
import pandas as pd 

with open('results_zwsichentest/results_1.csv', 'r') as f:
    reader = csv.reader(f)
    data = list(reader) 
df = pd.DataFrame(data[1:], columns=data[0])

# change names of the keys to be more readable
df.rename(columns={
    'Bewertung/20,00': 'Total', 
    'F 1 /3,00': 'F1', 
    'F 2 /1,00': 'F2',
    'F 3 /1,00': 'F3', 
    'F 4 /1,00': 'F4',
    'F 5 /3,00': 'F5',
    'F 6 /3,00': 'F6',
    'F 7 /1,00': 'F7',
    'F 8 /3,00': 'F8',
    'F 9 /4,00': 'F9', 
    }, inplace=True)

for i in range(1,10):
    df[f'F{i}'] = df[f'F{i}'].str.replace('-', '0.0').astype(str)
    df[f'F{i}'] = df[f'F{i}'].str.replace(',', '.').astype(float)

# print(df['F3'].tolist())

# # make a barchart for each F1-F9, with how many people got how mony points in each F1-F9, with the x-axis being the points and the y-axis being the frequency of people who got that many points
# for i in range(1,10):
#     plt.bar(df[f'F{i}'].value_counts().index, df[f'F{i}'].value_counts().values)
#     plt.title(f'Bar Chart of F{i}')
#     plt.xlabel(f'F{i} Score')
#     plt.ylabel('Frequency')
#     plt.show()

# --- candle bar plot --- #
means = []
stds = []
for i in range(1,10):
    means.append(df[f'F{i}'].mean())
    stds.append(df[f'F{i}'].std())
plt.errorbar(range(1,10), means, yerr=stds, fmt='o')
plt.title('Mean and Standard Deviation of F1-F9')
plt.xlabel('F1-F9')
plt.ylabel('Mean and Standard Deviation')
plt.xticks(range(1,10))
plt.show()  

# --- violin plot --- #
data = [df[f'F{i}'] for i in range(1,10)]
plt.violinplot(data, showmeans=True)
plt.title('Violin Plot of F1-F9')
plt.xlabel('F1-F9')
plt.ylabel('Scores')
plt.xticks(range(1,10))
plt.show()

# ---- histogram of Fi ---- #
i = 9

plt.hist(df[f'F{i}'], bins=int(max(df[f'F{i}'])), edgecolor='black')
plt.title(f'Histogram of F{i}')
plt.xlabel(f'F{i} Score')
plt.ylabel('Frequency')
plt.show()

print(df[f'F{i}'].tolist())