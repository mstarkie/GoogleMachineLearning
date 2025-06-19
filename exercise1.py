#@title Code - Load dependencies

#general

# machine learning
import  keras
# data
import numpy as np
import pandas as pd
import seaborn as sns
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
training_df['TRIP_MILES'].plot(kind='hist', bins=20, title='TRIP_MILES')
plt.gca().spines[['top', 'right',]].set_visible(False)
plt.show()
training_df['TRIP_SECONDS'].plot(kind='hist', bins=20, title='TRIP_SECONDS')
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