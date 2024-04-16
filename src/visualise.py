from typing import List, Dict, Any
from collections import defaultdict
import csv

from matplotlib import pyplot as plt
import pandas as pd
import numpy as np

def read_csv(file_path:str) -> List[dict]:
    """
    Reads the given path and returns a list of dicts.
    Dicts' keys are the header's names.

    Args:
        file_path (str)

    Returns:
        _type_: _description_
    """
    data = []
    with open(file_path, newline='', encoding='utf8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            data.append(row)
    return data

def generate_transposed_dict(csv_data:List[dict]) -> Dict[Any, list]:
    """Creates a Dict of lists from a List of Dicts for analytical purposes.

    Args:
        csv_data (List[dict])

    Returns:
        Dict[list]
    """

    headers = list(csv_data[0].keys())

    transposed = defaultdict(list)

    for attack_record in csv_data:
        for key in headers:
            transposed[key].append(attack_record[key])

    return dict(transposed)

def visualize_distribution(data):
    for key, values in data.items():
        if len(set(values)) < 50:
            plt.figure(figsize=(8, 6))
            plt.hist(values, bins=20, color='skyblue', edgecolor='black')
            plt.title(f"Distribution of values for key '{key}'")
            plt.xlabel("Value")
            plt.ylabel("Frequency")
            plt.show()
        else:
            print(key)

def calculate_zscores(df, col):
  mean = df[col].mean()
  std_dev = df[col].std()

  # Calculate Z-scores
  z_scores = (df[col] - mean) / std_dev

  return z_scores

def create_violin_plots(df, col, z_scores, threshold, labels, synth = False):
  # Plot the violin plot without the outliers
  plt.figure(figsize=(8, 6))

  # create a seperate plot for each class
  data = []
  for label in labels:
    data.append(df[(df['Type'] == label) & (z_scores <= threshold) & (z_scores >= -threshold)][col])
  # also for the whole dataset
  data.append(df[(z_scores <= threshold) & (z_scores >= -threshold)][col])

  # plot
  parts = plt.violinplot(data, showmeans = True, showextrema=True, showmedians=True)

  # Set the colors for the violins based on the category
  colors = ['Blue', 'Green', 'Purple', 'salmon']

  # Set the color of the violin patches
  for pc, color in zip(parts['bodies'], colors):
      pc.set_facecolor(color)

  # Create legend labels and handles for each violin plot
  legend_labels = list(labels)
  legend_labels.append('All data')
  legend_handles = [plt.Rectangle((0,0), 1, 1, color=color, edgecolor='black') for color in colors]

  # Plot settings
  plt.legend(legend_handles, legend_labels)
  plt.title(f'Violin plot of {col} by label')
  plt.ylabel('Values')
  plt.xlabel('Density')
  plt.grid(True)

  # decide what df is used
  if '' in df:
    df_type = 'attack'
  else:
    df_type = 'vector'

  if synth:
    plt.savefig(f"../imgs/synth_{df_type}_df_violin_{col}.jpg", bbox_inches='tight')
  else:
    plt.savefig(f"../imgs/{df_type}_df_violin_{col}.jpg", bbox_inches='tight')

  plt.show()