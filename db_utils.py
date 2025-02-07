import yaml
import psycopg2
import pandas as pd
from sqlalchemy import create_engine
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy import stats


def load_credentials(file_path='credentials.yaml'):
    """Loads database credentials from YAML file."""
    with open(file_path, 'r') as file:
        return yaml.safe_load(file)


class RDSDatabaseConnector:
    def __init__(self, credentials: dict):
        """Initializes database connection parameters."""
        self.host = credentials.get('RDS_HOST')
        self.user = credentials.get('RDS_USER')
        self.password = credentials.get('RDS_PASSWORD')
        self.dbname = credentials.get('RDS_DATABASE')
        self.port = str(credentials.get('RDS_PORT', 5432))
        self.engine = None

    def init_SQL_engine(self):
        """Creates a SQLAlchemy engine for database interaction."""
        connection_string = (
            f'postgresql+psycopg2://{self.user}:{self.password}@{self.host}:{self.port}/{self.dbname}'
        )
        self.engine = create_engine(connection_string)

    def fetch_data(self, query='SELECT * FROM failure_data'):
        """Fetches data from the database."""
        with self.engine.connect() as connection:
            return pd.read_sql(query, connection)

    def save_to_csv(self, data, filename="failure_data.csv"):
        """Saves DataFrame to CSV."""
        data.to_csv(filename, index=False)


class DataFrameProcessor:
    def __init__(self, df: pd.DataFrame):
        self.df = df

    def convert_to_numeric(self, column: str):
        self.df[column] = pd.to_numeric(self.df[column], errors='coerce')

    def convert_to_category(self, column: str):
        self.df[column] = self.df[column].astype('category')

    def count_nulls(self):
        null_counts = self.df.isnull().sum()
        null_percentage = (null_counts / len(self.df)) * 100
        return pd.DataFrame({'count': null_counts, 'percentage': null_percentage})

    def drop_columns_with_nulls(self, threshold=50):
        """Drops columns where missing values exceed the threshold."""
        to_drop = self.df.columns[self.df.isnull().mean() * 100 > threshold]
        self.df.drop(columns=to_drop, inplace=True)
        return to_drop

    def impute_nulls(self, strategy='mean'):
        """Imputes missing values using mean or median."""
        numeric_columns = self.df.select_dtypes(include=['number']).columns
        if strategy == 'mean':
            self.df[numeric_columns] = self.df[numeric_columns].fillna(self.df[numeric_columns].mean())
        elif strategy == 'median':
            self.df[numeric_columns] = self.df[numeric_columns].fillna(self.df[numeric_columns].median())

    def identify_skewed_columns(self, threshold=1.0):
        """Identifies skewed numeric columns (excluding binary ones)."""
        numeric_columns = self.df.select_dtypes(include=['number']).columns
        non_binary_columns = [col for col in numeric_columns if self.df[col].nunique() > 2]
        skewness = self.df[non_binary_columns].skew()
        return skewness[abs(skewness) > threshold].index

    def reduce_skew(self, skewed_columns):
        """Applies log or Box-Cox transformation to reduce skew."""
        for column in skewed_columns:
            if (self.df[column] <= 0).any():
                self.df[column] = np.log1p(self.df[column])  # Log transformation
            else:
                self.df[column], _ = stats.boxcox(self.df[column].dropna())

    def remove_outliers(self, column: str):
        """Removes outliers using IQR method."""
        Q1, Q3 = self.df[column].quantile([0.25, 0.75])
        IQR = Q3 - Q1
        lower, upper = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
        self.df = self.df[(self.df[column] >= lower) & (self.df[column] <= upper)]

    def remove_highly_correlated_columns(self, threshold=0.85):
        """Removes highly correlated numeric columns."""
        numeric_df = self.df.select_dtypes(include=['number'])
        if numeric_df.empty:
            print("No numeric columns available for correlation calculation.")
            return

        corr_matrix = numeric_df.corr().abs()
        upper_triangle = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        to_drop = [column for column in upper_triangle.columns if any(upper_triangle[column] > threshold)]
        self.df.drop(columns=to_drop, inplace=True)

    def save_transformed_data(self, filename='transformed_data.csv'):
        """Save the transformed dataframe to a CSV file."""
        self.df.to_csv(filename, index=False)
        print(f"Transformed data saved to {filename}")


class Plotter:
    def __init__(self, df: pd.DataFrame):
        self.df = df

    def plot_boxplots(self):
        numeric_columns = self.df.select_dtypes(include=['number']).columns
        self.df[numeric_columns].plot(kind='box', subplots=True, layout=(4, 3), figsize=(15, 20))
        plt.show()

    def plot_correlation_matrix(self):
        """Plots correlation heatmap."""
        plt.figure(figsize=(10, 8))
        sns.heatmap(self.df.corr(), annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
        plt.title("Correlation Matrix")
        plt.show()

# Load credentials and initialize database connection
credentials = load_credentials()
connector = RDSDatabaseConnector(credentials)
connector.init_SQL_engine()

# Fetch data and save to CSV
df = connector.fetch_data()
connector.save_to_csv(df, "failure_data.csv")

# Process data
processor = DataFrameProcessor(df)
processor.impute_nulls(strategy='mean')
skewed_columns = processor.identify_skewed_columns()
processor.reduce_skew(skewed_columns)
processor.remove_outliers(column='Rotational speed [rpm]')
processor.remove_outliers(column='Torque [Nm]')
processor.remove_highly_correlated_columns()

# Save the transformed dataframe
processor.save_transformed_data("transformed_data.csv")

