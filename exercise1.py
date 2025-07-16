#@title Copyright 2023 Google LLC. Double-click here for license information.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

#@title Code - Load dependencies

#general
import io

# machine learning
import  keras
# data
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
# data visualization
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from matplotlib import pyplot as plt
import seaborn as sns

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
print(training_df.describe(include='all'))
'''
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

# Part 2 - Dataset Exploration (ask questions)

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
#sns.pairplot(training_df, x_vars=["FARE"], y_vars=["FARE"], kind='scatter')
#plt.show()
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

# Part 3 - Train Model

#@title Define plotting functions

def make_plots(df, feature_names, label_name, model_output, sample_size=200):

  random_sample = df.sample(n=sample_size).copy()
  random_sample.reset_index()
  weights, bias, epochs, rmse = model_output

  is_2d_plot = len(feature_names) == 1
  model_plot_type = "scatter" if is_2d_plot else "surface"
  fig = make_subplots(rows=1, cols=2,
                      subplot_titles=("Loss Curve", "Model Plot"),
                      specs=[[{"type": "scatter"}, {"type": model_plot_type}]])

  plot_data(random_sample, feature_names, label_name, fig)
  plot_model(random_sample, feature_names, weights, bias, fig)
  plot_loss_curve(epochs, rmse, fig)

  fig.show()
  return

def plot_loss_curve(epochs, rmse, fig):
  curve = px.line(x=epochs, y=rmse)
  curve.update_traces(line_color='#ff0000', line_width=3)

  fig.append_trace(curve.data[0], row=1, col=1)
  fig.update_xaxes(title_text="Epoch", row=1, col=1)
  fig.update_yaxes(title_text="Root Mean Squared Error", row=1, col=1, range=[rmse.min()*0.8, rmse.max()])

  return

def plot_data(df, features, label, fig):
  if len(features) == 1:
    scatter = px.scatter(df, x=features[0], y=label)
  else:
    scatter = px.scatter_3d(df, x=features[0], y=features[1], z=label)

  fig.append_trace(scatter.data[0], row=1, col=2)
  if len(features) == 1:
    fig.update_xaxes(title_text=features[0], row=1, col=2)
    fig.update_yaxes(title_text=label, row=1, col=2)
  else:
    fig.update_layout(scene1=dict(xaxis_title=features[0], yaxis_title=features[1], zaxis_title=label))

  return

def plot_model(df, features, weights, bias, fig):
  df['FARE_PREDICTED'] = bias[0]

  for index, feature in enumerate(features):
    df['FARE_PREDICTED'] = df['FARE_PREDICTED'] + weights[index][0] * df[feature]

  if len(features) == 1:
    model = px.line(df, x=features[0], y='FARE_PREDICTED')
    model.update_traces(line_color='#ff0000', line_width=3)
  else:
    z_name, y_name = "FARE_PREDICTED", features[1]
    z = [df[z_name].min(), (df[z_name].max() - df[z_name].min()) / 2, df[z_name].max()]
    y = [df[y_name].min(), (df[y_name].max() - df[y_name].min()) / 2, df[y_name].max()]
    x = []
    for i in range(len(y)):
      x.append((z[i] - weights[1][0] * y[i] - bias[0]) / weights[0][0])

    plane=pd.DataFrame({'x':x, 'y':y, 'z':[z] * 3})

    light_yellow = [[0, '#89CFF0'], [1, '#FFDB58']]
    model = go.Figure(data=go.Surface(x=plane['x'], y=plane['y'], z=plane['z'],
                                      colorscale=light_yellow))

  fig.add_trace(model.data[0], row=1, col=2)

  return

def model_info(feature_names, label_name, model_output):
  weights = model_output[0]
  bias = model_output[1]

  nl = "\n"
  header = "-" * 80
  banner = header + nl + "|" + "MODEL INFO".center(78) + "|" + nl + header

  info = ""
  equation = label_name + " = "

  for index, feature in enumerate(feature_names):
    info = info + "Weight for feature[{}]: {:.3f}\n".format(feature, weights[index][0])
    equation = equation + "{:.3f} * {} + ".format(weights[index][0], feature)

  info = info + "Bias: {:.3f}\n".format(bias[0])
  equation = equation + "{:.3f}\n".format(bias[0])

  return banner + nl + info + nl + equation

print("SUCCESS: defining plotting functions complete.")

#@title Code - Define ML functions

def build_model(my_learning_rate, num_features):
  """Create and compile a simple linear regression model."""
  # Describe the topography of the model.
  # The topography of a simple linear regression model
  # is a single node in a single layer.
  inputs = keras.Input(shape=(num_features,))
  outputs = keras.layers.Dense(units=1)(inputs)
  model = keras.Model(inputs=inputs, outputs=outputs)

  # Compile the model topography into code that Keras can efficiently
  # execute. Configure training to minimize the model's mean squared error.
  model.compile(optimizer=keras.optimizers.RMSprop(learning_rate=my_learning_rate),
                loss="mean_squared_error",
                metrics=[keras.metrics.RootMeanSquaredError()])

  return model


def train_model(model, features, label, epochs, batch_size):
  """Train the model by feeding it data."""

  # Feed the model the feature and the label.
  # The model will train for the specified number of epochs.
  history = model.fit(x=features,
                      y=label,
                      batch_size=batch_size,
                      epochs=epochs)

  # Gather the trained model's weight and bias.
  trained_weight = model.get_weights()[0]
  trained_bias = model.get_weights()[1]

  # The list of epochs is stored separately from the rest of history.
  epochs = history.epoch

  # Isolate the error for each epoch.
  hist = pd.DataFrame(history.history)

  # To track the progression of training, we're going to take a snapshot
  # of the model's root mean squared error at each epoch.
  rmse = hist["root_mean_squared_error"]

  return trained_weight, trained_bias, epochs, rmse


def run_experiment(df, feature_names, label_name, learning_rate, epochs, batch_size):

  print('INFO: starting training experiment with features={} and label={}\n'.format(feature_names, label_name))

  num_features = len(feature_names)

  features = df.loc[:, feature_names].values
  label = df[label_name].values

  model = build_model(learning_rate, num_features)
  model_output = train_model(model, features, label, epochs, batch_size)

  print('\nSUCCESS: training experiment complete\n')
  print('{}'.format(model_info(feature_names, label_name, model_output)))
  make_plots(df, feature_names, label_name, model_output)

  return model

print("SUCCESS: defining linear regression functions complete.")

#@title Code - Experiment 1

# The following variables are the hyperparameters.

# Learning Rate controls how big a step the algorithm takes in the direction of the negative gradience at each iteration.
#   When the learning rate is too high, the loss curve bounces around and does not
#   appear to be moving towards convergence with each iteration. Also, notice that
#   the predicted model does not fit the data very well. With a learning rate that
#   is too high, it is unlikely that you will be able to train a model with good
#   results.
#
#   When the learning rate is too small, it may take longer for the loss curve to
#   converge. With a small learning rate the loss curve decreases slowly, but does
#   not show a dramatic drop or leveling off. With a small learning rate you could
#   increase the number of epochs so that your model will eventually converge, but
#   it will take longer.

# Batch Size: Number of training samples used in one iteration before updating the model's weights.
#     batch_size = 1: Stochastic Gradient Descent
#     batch_size = total samples:  Batch Gradient Descent
#     1 < batch_size < total:  Mini-Batch Gradient Descent.
#     decreasing:  Adds noise (unstable convergence), better generalization, updates are fast,  takes longer,
#
#     increasing:  More stable updates, fewer iterations per epoch, may converge on a sharp minima
#                  May require higher learning rates to maintain momentum.
#                  Increasing the batch size makes each epoch run faster, but as with the smaller
#                  learning rate, the model does not converge with just 20 epochs. If you have
#                  time, try increasing the number of epochs.  Eventually you should see the
#                  model converge.

# Epoch:  1 pass through the dataset.  Too few: underfitting, poor learning.  Too many: Overfitting, loss increases.
#    Stop when loss starts to increase.


learning_rate = 0.001
epochs = 20
batch_size = 50

# Specify the feature and the label.
features = ['TRIP_MILES']
label = 'FARE'

model_1 = run_experiment(training_df, features, label, learning_rate, epochs, batch_size)

#@title Double-click to view answers for training model with one feature

print("How many epochs did it take to converge on the final model?")
# -----------------------------------------------------------------------------
answer = """
Use the loss curve to see where the loss begins to level off during training.

With this set of hyperparameters:

  learning_rate = 0.001
  epochs = 20
  batch_size = 50

it takes about 5 epochs for the training run to converge to the final model.
"""
print(answer)

print("How well does the model fit the sample data?")
# -----------------------------------------------------------------------------
answer = '''
It appears from the model plot that the model fits the sample data fairly well.
'''
print(answer)

training_df['TRIP_MINUTES'] = training_df['TRIP_SECONDS']/60
features = ['TRIP_MINUTES']
model_2 = run_experiment(training_df, features, label, learning_rate, epochs, batch_size)

features = ['TRIP_MILES', 'TRIP_MINUTES']
model_3 = run_experiment(training_df, features, label, learning_rate, epochs, batch_size)

#@title Double-click to view answers for training with two features

print("Does the model with two features produce better results than one using a single feature?")
# -----------------------------------------------------------------------------
answer = '''
To answer this question for your specific training runs, compare the RMSE for
each model. For example, if the RMSE for the model trained with one feature was
3.7457 and the RMSE for the model with two features is 3.4787, that means that
on average the model with two features makes predictions that are about $0.27
closer to the observed fare.

'''
print(answer)

print("Does it make a difference if you use TRIP_SECONDS instead of TRIP_MINUTES?")
# -----------------------------------------------------------------------------
answer = '''
When training a model with more than one feature, it is important that all
numeric values are roughly on the same scale. In this case, TRIP_SECONDS and
TRIP_MILES do not meet this criteria. The mean value for TRIP_MILES is 8.3 and
the mean for TRIP_SECONDS is 1,320; that is two orders of magnitude difference.
In contrast, the mean for TRIP_MINUTES is 22, which is more similar to the scale
of TRIP_MILES (8.3) than TRIP_SECONDS (1,320). Of course, this is not the
only way to scale values before training, but you will learn about that in
another module.
'''
print(answer)

print("How well do you think the model comes to the ground truth fare calculation for Chicago taxi trips?")
# -----------------------------------------------------------------------------
answer = '''
In reality, Chicago taxi cabs use a documented formula to determine cab fares.
For a single passenger paying cash, the fare is calculated like this:

FARE = 2.25 * TRIP_MILES + 0.12 * TRIP_MINUTES + 3.25

Typically with machine learning problems you would not know the 'correct'
formula, but in this case you can use this knowledge to evaluate your model.
Take a look at your model output (the weights and bias) and determine how
well it matches the ground truth fare calculation. You should find that the
model is roughly close to this formula.
'''
print(answer)

#@title Code - Define functions to make predictions
def format_currency(x):
  return "${:.2f}".format(x)

def build_batch(df, batch_size):
  batch = df.sample(n=batch_size).copy()
  batch.set_index(np.arange(batch_size), inplace=True)
  return batch

def predict_fare(model, df, features, label, batch_size=50):
  batch = build_batch(df, batch_size)
  predicted_values = model.predict_on_batch(x=batch.loc[:, features].values)

  data = {"PREDICTED_FARE": [], "OBSERVED_FARE": [], "L1_LOSS": [],
          features[0]: [], features[1]: []}
  for i in range(batch_size):
    predicted = predicted_values[i][0]
    observed = batch.at[i, label]
    data["PREDICTED_FARE"].append(format_currency(predicted))
    data["OBSERVED_FARE"].append(format_currency(observed))
    data["L1_LOSS"].append(format_currency(abs(observed - predicted)))
    data[features[0]].append(batch.at[i, features[0]])
    data[features[1]].append("{:.2f}".format(batch.at[i, features[1]]))

  output_df = pd.DataFrame(data)
  return output_df

def show_predictions(output):
  header = "-" * 80
  banner = header + "\n" + "|" + "PREDICTIONS".center(78) + "|" + "\n" + header
  print(banner)
  print(output)
  return

#@title Code - Make predictions

output = predict_fare(model_3, training_df, features, label)
show_predictions(output)

print("How close is the predicted value to the label value?")
# -----------------------------------------------------------------------------
answer = '''
Based on a random sampling of examples, the model seems to do pretty well
predicting the fare for a taxi ride. Most of the predicted values do not vary
significantly from the observed value. You should be able to see this by looking
at the column L1_LOSS = |observed - predicted|.
'''
print(answer)
