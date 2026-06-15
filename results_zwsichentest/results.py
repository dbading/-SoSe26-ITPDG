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

# give me all keys in the dataframe
# print(df.keys())

# check for missing values in the dataframe
# print(df.isnull().sum())

# print list of values to 'Total' column
# print(df['Total'].tolist())

# # # convert the string in the dataframe to a float, replace the comma with a dot
# df['Total'] = df['Total'].str.replace(',', '.').astype(float)
# df['F1'] = df['F1'].str.replace(',', '.').astype(float)
# df['F2'] = df['F2'].str.replace(',', '.').astype(float)
# df['F3'] = df['F3'].str.replace('-', '0.0').astype(str)
# df['F3'] = df['F3'].str.replace(',', '.').astype(float)
# df['F4'] = df['F4'].str.replace(',', '.').astype(float)
# df['F5'] = df['F5'].str.replace('-', '0.0').astype(str)
# df['F5'] = df['F5'].str.replace(',', '.').astype(float)
# df['F6'] = df['F6'].str.replace('-', '0.0').astype(str)
# df['F6'] = df['F6'].str.replace(',', '.').astype(float)
# df['F7'] = df['F7'].str.replace('-', '0.0').astype(str)
# df['F7'] = df['F7'].str.replace(',', '.').astype(float)
# df['F8'] = df['F8'].str.replace('-', '0.0').astype(str)
# df['F8'] = df['F8'].str.replace(',', '.').astype(float)
# df['F9'] = df['F9'].str.replace('-', '0.0').astype(str)
# df['F9'] = df['F9'].str.replace(',', '.').astype(float)

for i in range(1,10):
    df[f'F{i}'] = df[f'F{i}'].str.replace('-', '0.0').astype(str)
    df[f'F{i}'] = df[f'F{i}'].str.replace(',', '.').astype(float)

# print(df['F3'].tolist())

# ---- histogram of F1 ---- #

# print(df['F3'].tolist())

i = 5

plt.hist(df[f'F{i}'], bins=int(max(df[f'F{i}'])), edgecolor='black')
plt.title(f'Histogram of F{i}')
plt.xlabel(f'F{i} Score')
plt.ylabel('Frequency')
plt.show()

# distribution of all F1-F9 in one plot
plt.figure(figsize=(10,6))
for i in range(1,10): # now we like to have a distribution as plt.plot for each F1-F9 in one plot
    plt.plot(df[f'F{i}'], label=f'F{i}')
plt.title('Distribution of F1-F9')
plt.xlabel('Index')
plt.ylabel('Score')
plt.legend()
plt.show()  