# Analysis on Automobile DataSet

This repository contains a data analysis project on the automobile dataset from the UCI Machine Learning Repository. The analysis involves data cleaning, handling missing values, and examining correlations between different attributes of the dataset. The analysis is performed in the Jupyter notebook: `Data Analysis on Automobile dataset.ipynb`.

## Table of Contents

- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Code Overview](#code-overview)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Dataset

The dataset used in this project is related to the fuel consumption and carbon dioxide emissions of automobiles. The data can be accessed from the UCI Machine Learning Repository: [Automobile Data Set](https://archive.ics.uci.edu/ml/machine-learning-databases/autos/imports-85.data).

## Installation

To run the notebook, you need to have Python installed along with the following libraries:

- pandas
- numpy
- seaborn
- matplotlib

You can install these libraries using pip:

## Code Overview

### Data Loading and Cleaning

The data is loaded from the UCI repository and cleaned by replacing missing values with the mean of their respective columns.

```python
import pandas as pd
import numpy as np

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/autos/imports-85.data"
df = pd.read_csv(url, header=None)
```

### Handling Missing Values
Missing values in columns such as normalized-losses, bore, stroke, horsepower, and peak-rpm are replaced with the mean values of their respective columns.
```
df.replace("?", np.nan, inplace=True)
df["normalized-losses"] = pd.to_numeric(df["normalized-losses"], errors='coerce')
avg_norm_losses = df["normalized-losses"].astype(float).mean(axis=0)
df["normalized-losses"] = df["normalized-losses"].fillna(avg_norm_losses)
```

### Data Transformation and Normalization
The data is further processed by dropping rows with missing price values and normalizing the length column.
```
df = df.dropna(subset=["price"], axis=0)
df["length"] = df["length"] / df["length"].max()
```

### Exploratory Data Analysis
Correlation between different columns is analyzed, and visualizations are created using seaborn and matplotlib.
```
import seaborn as sns
import matplotlib.pyplot as plt

sns.regplot(x="engine-size", y="price", data=df)
plt.ylim(0,)
```


