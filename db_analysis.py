import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from db_utils import DataFrameProcessor, Plotter


df = pd.read_csv('failure_data.csv')

#Impute nulls
processor = DataFrameProcessor(df)
processor.impute_nulls()
df = processor.df

#Calculate and display operating ranges
columns_of_interest = [
    'Air temperature [K]', 
    'Process temperature [K]', 
    'Rotational speed [rpm]', 
    'Torque [Nm]', 
    'Tool wear [min]'
]

range_table = df[columns_of_interest].agg(['min', 'max', 'mean']).transpose()

print('Operating ranges:')
print(range_table)

# Calculate and display operating ranges grouped by type
grouped_ranges = df.groupby('Type')[columns_of_interest].agg(['min', 'max', 'mean'])

print("Operating Ranges by Product Quality Type:")
print(grouped_ranges)

#Calculate and visualise upper limits of Tool Wear
plt.figure(figsize=(10, 6))
sns.histplot(df['Tool wear [min]'], bins=30, kde=True, color='blue')

plt.xlabel("Tool Wear (Minutes)")
plt.ylabel("Number of Tools")
plt.title("Distribution of Tool Wear Values")
plt.show()


#Calculate failures
failure_columns = ['TWF', 'HDF', 'PWF', 'OSF', 'RNF', 'Machine failure']

failure_counts = df[failure_columns].sum()

total_failures = failure_counts.sum()
total_records = len(df)
failure_percentage = (total_failures / total_records) * 100

print("Failure Counts:\n", failure_counts)
print(f"\nTotal Failures: {total_failures} out of {total_records} records")
print(f"Percentage of failures: {failure_percentage:.2f}%")

#Calculate failures by quality type
failures_by_quality = df.groupby('Type')[failure_columns].sum()

failures_by_quality_percentage = failures_by_quality.div(failures_by_quality.sum(axis=1), axis=0) * 100
failures_by_quality_percentage = failures_by_quality_percentage.round(2)

print("\nFailures by Product Quality Type (absolute count):\n", failures_by_quality)
print("\nFailures by Product Quality Type (percentage):\n", failures_by_quality_percentage)

# Calculate and visualise leading causes of failure
total_failure_by_cause = df[failure_columns].sum()

sorted_failure_by_cause = total_failure_by_cause.sort_values(ascending=False)

print("\nLeading Causes of Failure:\n", sorted_failure_by_cause)

plt.figure(figsize=(10, 6))
sns.barplot(x=sorted_failure_by_cause.index, y=sorted_failure_by_cause.values)

plt.xlabel("Failure Cause")
plt.ylabel("Number of Failures")
plt.title("Leading Causes of Failure in the Manufacturing Process")
plt.show()

#Remove outliers
processor.remove_outliers(column='Rotational speed [rpm]')
processor.remove_outliers(column='Torque [Nm]')

#Calculate correlation between settings and failures
settings = ['Air temperature [K]', 'Process temperature [K]', 'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]']
correlations = df[settings + ['Machine failure']].corr()
print(correlations['Machine failure'].sort_values(ascending=False))

failure_types = ['TWF', 'HDF', 'PWF', 'OSF', 'RNF']
for failure in failure_types:
    print(f"Correlation of {failure} with machine settings:")
    print(df[settings + [failure]].corr()[failure].sort_values(ascending=False), "\n")

#Calculate probability of failure for each range of operation
bin_ranges = {
    "Torque [Nm]": [0, 20, 40, 60, 80],  
    "Rotational speed [rpm]": [1000, 1500, 2000, 2500, 3000],
    "Tool wear [min]": [0, 50, 100, 150, 200, 250],
    "Air temperature [K]": [295, 298, 301, 304, 307],
    "Process temperature [K]": [305, 308, 311, 314, 317]
}

failure_columns = ["TWF", "HDF", "PWF", "OSF", "RNF", "Machine failure"]
df["Failure Occurred"] = df[failure_columns].max(axis=1) 

plt.figure(figsize=(15, 10))

for i, (col, bins) in enumerate(bin_ranges.items()):
    plt.subplot(2, 3, i + 1)
    
    df[f"{col} bin"] = pd.cut(df[col], bins=bins)
    
    failure_rates = df.groupby(f"{col} bin", observed=True)["Failure Occurred"].mean() * 100
    
    # Plot bar chart
    sns.barplot(x=failure_rates.index.astype(str), y=failure_rates.values, palette="Blues_r")
    
    plt.xticks(rotation=45, ha="right")  
    plt.xlabel(f"{col} Range")
    plt.ylabel("Failure Percentage (%)")
    plt.title(f"Failure Rate vs {col}")

plt.tight_layout()
plt.show()
