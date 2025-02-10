DataFrame Processor and Failure Analysis
Overview

This project provides a set of tools to process, analyze, and visualize manufacturing data to identify risk factors for machine failure. The main goal is to clean the data, explore relationships between various settings and failures, and help optimize machine settings to prevent failures.

The project includes a DataFrameProcessor class that can clean and transform data by handling missing values, removing highly correlated columns, and saving the cleaned data to a CSV file.
Features

    Data Preprocessing:
        Handle missing values.
        Convert data types (e.g., numeric, categorical).
        Remove outliers.

    Correlation Handling:
        Identify and remove highly correlated columns to improve data quality and avoid multicollinearity.

    Failure Analysis:
        Calculate failure rates for different failure types.
        Visualize failure distributions by product quality type and machine settings.

    Data Transformation and Saving:
        Clean and transform the data.
        Save the transformed DataFrame to a CSV file for further analysis.

Installation

To get started, clone the repository and install the required dependencies:

    Clone the repository:

git clone https://github.com/your-username/your-project-name.git

    Install the dependencies:

pip install pandas numpy seaborn matplotlib

Usage
Import the Class

Start by importing the necessary classes from the dataframe_processor module:

from dataframe_processor import DataFrameProcessor

Load Your Data

Load your dataset into a pandas DataFrame:

import pandas as pd

# Load your data
df = pd.read_csv('your_data.csv')

Initialize the Processor

Create an instance of the DataFrameProcessor by passing your pandas DataFrame:

# Initialize the processor
processor = DataFrameProcessor(df)

Handle Missing Values

The processor will impute missing values using a default strategy or a custom one:

# Impute missing values
processor.impute_nulls()
df = processor.df

Remove Highly Correlated Columns

To remove columns with high correlations, you can set a threshold. For example, if the correlation between two columns exceeds 0.85, they will be removed:

# Remove highly correlated columns
processor.remove_highly_correlated_columns(threshold=0.85)

Save Transformed Data

After processing, you can save the cleaned DataFrame to a CSV file for further analysis:

# Save the transformed data
processor.save_transformed_data(filename='transformed_data.csv')

Failure Analysis

The code includes steps to analyze the failure patterns across different machine settings. It calculates failure rates, displays distributions of failures, and plots bar charts for each machine setting showing the failure percentage across different ranges of settings.

# Example of failure analysis code
failure_columns = ['TWF', 'HDF', 'PWF', 'OSF', 'RNF', 'Machine failure']
df["Failure Occurred"] = df[failure_columns].max(axis=1)

# Calculate failure rates for different settings
bin_ranges = {
    "Torque [Nm]": [0, 20, 40, 60, 80],  
    "Rotational speed [rpm]": [1000, 1500, 2000, 2500, 3000],
    "Tool wear [min]": [0, 50, 100, 150, 200, 250],
    "Air temperature [K]": [295, 298, 301, 304, 307],
    "Process temperature [K]": [305, 308, 311, 314, 317]
}

# Visualize failure rate
for col, bins in bin_ranges.items():
    df[f"{col} bin"] = pd.cut(df[col], bins=bins)
    failure_rates = df.groupby(f"{col} bin")["Failure Occurred"].mean() * 100
    sns.barplot(x=failure_rates.index.astype(str), y=failure_rates.values)

Example

Here’s a full example of how to use the DataFrameProcessor class to clean the data, remove highly correlated columns, and save the cleaned data:

import pandas as pd
from dataframe_processor import DataFrameProcessor

# Load your data
df = pd.read_csv('your_data.csv')

# Initialize the processor
processor = DataFrameProcessor(df)

# Handle missing values
processor.impute_nulls()
df = processor.df

# Remove highly correlated columns
processor.remove_highly_correlated_columns(threshold=0.85)

# Save the transformed data to a CSV
processor.save_transformed_data(filename='transformed_data.csv')

