#@title Code - Load dependencies

#general

# machine learning
import  keras
# data
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
# data visualization
from plotly.subplots import make_subplots
from matplotlib import pyplot as plt


# @title
# The following code cell loads the dataset and creates a pandas DataFrame.
# You can think of a DataFrame like a spreadsheet with rows and columns.
# The rows represent individual data examples, and the columns represent the attributes associated with each example.
chicago_taxi_dataset = pd.read_csv("https://download.mlcc.google.com/mledu-datasets/chicago_taxi_train.csv")

# Updates dataframe to use specific columns.
training_df = chicago_taxi_dataset[['TRIP_MILES', 'TRIP_SECONDS', 'FARE', 'COMPANY', 'PAYMENT_TYPE', 'TIP_RATE']]
pd.set_option('display.width', 400)
pd.set_option('display.max_columns', None)
print('Read dataset completed successfully.')
print('Total number of rows: {0}\n\n'.format(len(training_df.index)))
print(training_df.head(200))

#Dataset Exploration (view statistics)
'''
print(training_df.describe(include='all'))
training_df['TRIP_SECONDS'].plot(kind='hist', bins=20, title='TRIP_SECONDS')
plt.gca().spines[['top', 'right',]].set_visible(False)
plt.show()

training_df['TRIP_MILES'].plot(kind='hist', bins=20, title='TRIP_MILES')
plt.gca().spines[['top', 'right',]].set_visible(False)
plt.show()

training_df['FARE'].plot(kind='hist', bins=20, title='FARE')
plt.gca().spines[['top', 'right',]].set_visible(False)
plt.show()

training_df['TIP_RATE'].plot(kind='hist', bins=20, title='TIP_RATE')
plt.gca().spines[['top', 'right',]].set_visible(False)
plt.show()

training_df.groupby('PAYMENT_TYPE').size().plot(kind='barh', color=sns.palettes.mpl_palette('Dark2'), title='PAYMENT_TYPE')
plt.gca().spines[['top', 'right',]].set_visible(False)
plt.show()

training_df.plot(kind='scatter', x='TRIP_MILES', y='TRIP_SECONDS', s=32, alpha=.8, title='TRIP_MILES vs TRIP_SECONDS')
plt.gca().spines[['top', 'right',]].set_visible(False)
plt.show()

training_df.plot(kind='scatter', x='TRIP_SECONDS', y='FARE', s=32, alpha=.8)
plt.gca().spines[['top', 'right',]].set_visible(False)
plt.show()

training_df.plot(kind='scatter', x='FARE', y='TIP_RATE', s=32, alpha=.8)
plt.gca().spines[['top', 'right',]].set_visible(False)
plt.show()

training_df['TRIP_MILES'].plot(kind='line', figsize=(8, 4), title='TRIP_MILES')
plt.gca().spines[['top', 'right']].set_visible(False)
plt.show()

training_df['TRIP_SECONDS'].plot(kind='line', figsize=(8, 4), title='TRIP_SECONDS')
plt.gca().spines[['top', 'right']].set_visible(False)
plt.show()

training_df['FARE'].plot(kind='line', figsize=(8, 4), title='FARE')
plt.gca().spines[['top', 'right']].set_visible(False)
plt.show()

training_df['TIP_RATE'].plot(kind='line', figsize=(8, 4), title='TIP_RATE')
plt.gca().spines[['top', 'right']].set_visible(False)
plt.show()

figsize = (12, 1.2 * len(training_df['PAYMENT_TYPE'].unique()))
plt.figure(figsize=figsize)
sns.violinplot(training_df, x='TRIP_MILES', y='PAYMENT_TYPE', inner='box', palette='Dark2')
sns.despine(top=True, right=True, bottom=True, left=True)
plt.show()

figsize = (12, 1.2 * len(training_df['PAYMENT_TYPE'].unique()))
plt.figure(figsize=figsize)
sns.violinplot(training_df, x='TRIP_SECONDS', y='PAYMENT_TYPE', inner='box', palette='Dark2')
sns.despine(top=True, right=True, bottom=True, left=True)
plt.show()

figsize = (12, 1.2 * len(training_df['PAYMENT_TYPE'].unique()))
plt.figure(figsize=figsize)
sns.violinplot(training_df, x='FARE', y='PAYMENT_TYPE', inner='box', palette='Dark2')
sns.despine(top=True, right=True, bottom=True, left=True)
plt.show()

figsize = (12, 1.2 * len(training_df['PAYMENT_TYPE'].unique()))
plt.figure(figsize=figsize)
sns.violinplot(training_df, x='TIP_RATE', y='PAYMENT_TYPE', inner='box', palette='Dark2')
sns.despine(top=True, right=True, bottom=True, left=True)
plt.show()
'''

#Dataset Exploration (ask questions)

# You should be able to find the answers to the questions about the dataset
# by inspecting the table output after running the DataFrame describe method.
#
# Run this code cell to verify your answers.

# What is the maximum fare?
max_fare = training_df['FARE'].max()
print("What is the maximum fare? \t\t\t\tAnswer: ${fare:.2f}".format(fare = max_fare))

# What is the mean distance across all trips?
mean_distance = training_df['TRIP_MILES'].mean()
print("What is the mean distance across all trips? \t\tAnswer: {mean:.4f} miles".format(mean = mean_distance))

# How many cab companies are in the dataset?
num_unique_companies =  training_df['COMPANY'].nunique()
print("How many cab companies are in the dataset? \t\tAnswer: {number}".format(number = num_unique_companies))

# What is the most frequent payment type?
most_freq_payment_type = training_df['PAYMENT_TYPE'].value_counts().idxmax()
print("What is the most frequent payment type? \t\tAnswer: {type}".format(type = most_freq_payment_type))

# Are any features missing data (isnull = true/false matrix, sum() = sum for each column,
# sum().sum() = sum of all columns
missing_values = training_df.isnull().sum().sum()
print("Are any features missing data? \t\t\t\tAnswer:", "No" if missing_values == 0 else "Yes")

#View correlation matrix


desc = '''
a correlation matrix to identify features whose values correlate well with the label. 

Correlation values have the following meanings:
    1.0: perfect positive correlation; that is, when one attribute rises, the other attribute rises.
   -1.0: perfect negative correlation; that is, when one attribute rises, the other attribute falls.
    0.0: no correlation; the two columns are not linearly related.
In general, the higher the absolute value of a correlation value, the greater its predictive power.
'''

#print(desc)

training_df.corr(numeric_only = True)
#sns.pairplot(training_df, x_vars=["FARE", "TRIP_MILES", "TRIP_SECONDS"], y_vars=["FARE", "TRIP_MILES", "TRIP_SECONDS"])
sns.pairplot(training_df, x_vars=["FARE"], y_vars=["FARE"], kind='scatter')
plt.show()
# Which feature correlates most strongly to the label FARE?
# ---------------------------------------------------------
answer = '''
The feature with the strongest correlation to the FARE is TRIP_MILES.
As you might expect, TRIP_MILES looks like a good feature to start with to train
the model. Also, notice that the feature TRIP_SECONDS has a strong correlation
with fare too. 
'''
print(answer)


# Which feature correlates least strongly to the label FARE?
# -----------------------------------------------------------
answer = '''The feature with the weakest correlation to the FARE is TIP_RATE.'''
print(answer)