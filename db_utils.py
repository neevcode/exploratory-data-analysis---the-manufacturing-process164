import yaml
import psycopg2
import pandas as pd
from sqlalchemy import create_engine
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy import stats

#Loads database credentials from YAML file and returns them in dictionary format
def load_credentials(file_path='credentials.yaml'):
    print("Loading credentials...")
    with open(file_path, 'r') as file:
        credentials = yaml.safe_load(file)
    print("Credentials loaded:", credentials) 
    return credentials


class RDSDatabaseConnector:
    def __init__ (self, credentials: dict):
        print("Initializing RDSDatabaseConnector...") 
        self.host = credentials.get('RDS_HOST')
        self.user = credentials.get('RDS_USER')
        self.password = credentials.get('RDS_PASSWORD')
        self.dbname = credentials.get('RDS_DATABASE')
        self.port = str(credentials.get('RDS_PORT', 5432))
        self.engine = None

    def init_SQL_engine(self):
        print("Initializing SQLAlchemy engine...")
        connection_string = f'postgresql+psycopg2://{self.user}:{self.password}@{self.host}:{self.port}/{self.dbname}'
        print(f"Connecting to: postgresql+psycopg2://{self.user}:****@{self.host}:{self.port}/{self.dbname}")  # Debugging (masked password)
        
        self.engine = create_engine(connection_string)
        print("Database engine initialized successfully")

    def fetch_data(self):
        query = 'SELECT * FROM failure_data'
        with self.engine.connect() as connection:
            result = pd.read_sql(query, connection)
        return result

    def save_to_csv(self, data, filename="output_data.csv"):
        print(f"Saving data to {filename}...")
        data.to_csv(filename, index=False)
        print(f"Data successfully saved to {filename}")

def load_local_data(file_path='failure_data.csv'):
    return pd.read_csv(file_path)

#Transform data, change datatypes of columns
class DataTransform:
    def __init__ (self, df: pd.DataFrame):
        self.df = df

#Convert columns to numeric, will convert to NaN if incompatible
    def convert_to_numeric(self, column: str):
        self.df[column] = pd.to_numeric(self.df[column], errors='coerce')

#I THINK THIS COLUMN IS REDUNDANT, NO DATETIME COLUMNS IN DF
#Convert to datetime, will convert to NaN if incompatible
    def convert_to_datetime(self, column: str, date_format: str = None):
        """Convert a column to datetime format."""
        self.df[column] = pd.to_datetime(self.df[column], format=date_format, errors='coerce')

#Convert to categorial 
    def convert_to_category(self, column: str):
        self.df[column] = self.df[column].astype('category')


#Get descriptions of dataframe
class DataFrameInfo:
    def __init__(self, df: pd.DataFrame):
        self.df = df

    def describe_columns(self):
        return self.df.dtypes
    

###Issues with this function
    def statistical_summary(self):
        numeric_cols = self.df.select_dtypes(include=['number']) 
        summary = pd.DataFrame({
            'mean': numeric_cols.mean(),
            'std': numeric_cols.std(),
            'median': numeric_cols.median()
        })
    
    def count_distinct(self, column: str):
        return self.df[column].nunique()
    
    def shape(self):
        return self.df.shape
    
    def count_nulls(self):
        null_counts = self.df.isnull().sum()
        null_percentage = (null_counts / len(self.df)) * 100
        return pd.DataFrame({'count': null_counts, 'percentage': null_percentage})
    
    def unique_values(self):
        return self.df.nunique()
    
    def correlations(self):
        return self.df.corr()
    

class DataFrameTransform:
    def __init__(self, df: pd.DataFrame):
        self.df = df

    def count_nulls(self):
        null_counts = self.df.isnull().sum()
        null_percentage = (null_counts / len(self.df)) * 100
        return pd.DataFrame({'count': null_counts, 'percentage': null_percentage})

    def drop_columns_with_nulls(self, threshold=50):
        null_summary = self.count_nulls()
        columns_to_drop = null_summary[null_summary['percentage'] > threshold].index
        self.df.drop(columns=columns_to_drop, inplace=True)
        return columns_to_drop

    def impute_nulls(self, strategy='mean'):
        numeric_columns = self.df.select_dtypes(include=['float64', 'int64']).columns
        if strategy == 'mean':
            self.df[numeric_columns] = self.df[numeric_columns].fillna(self.df[numeric_columns].mean())
        elif strategy == 'median':
            self.df[numeric_columns] = self.df[numeric_columns].fillna(self.df[numeric_columns].median())
        else:
            raise ValueError("Strategy must be 'mean' or 'median'")

    def identify_skewed_columns(self, threshold=1.0):
        numeric_columns = self.df.select_dtypes(include=['number']).columns
    
    # Exclude binary columns (only 0 and 1 values)
        non_binary_columns = [
            col for col in numeric_columns 
            if self.df[col].nunique() > 2  
        ]
    
        skewness = self.df[non_binary_columns].skew()
        skewed_columns = skewness[abs(skewness) > threshold]

        if skewed_columns.empty:
            print("No skewed columns identified.")
        return skewed_columns.index

    def reduce_skew(self, skewed_columns):
        for column in skewed_columns:
            if (self.df[column] <= 0).any():
            # If column has non-positive values, apply log transformation
                print(f"Applying log transformation to {column} (Box-Cox not possible)")
                self.df[column] = np.log1p(self.df[column])
            else:
            # Try Box-Cox transformation
                transformed_data, lambda_val = stats.boxcox(self.df[column].dropna())  # Drop NaNs for Box-Cox
            
            # Check if transformation significantly reduces skew
                original_skew = self.df[column].skew()
                transformed_skew = pd.Series(transformed_data).skew()
            
                if abs(transformed_skew) < abs(original_skew):  
                    print(f"Applying Box-Cox to {column} (λ={lambda_val:.2f}), Skew reduced from {original_skew:.2f} to {transformed_skew:.2f}")
                # Replace original column with transformed values
                    self.df[column] = transformed_data
                else:
                    print(f"Box-Cox did not significantly reduce skew for {column}, keeping original values")

    def save_transformed_data(self, filename='transformed_data.csv'):
        self.df.to_csv(filename, index=False)

    def remove_outliers(self, column: str, method='IQR'):
            
        # Calculate Q1 (25th percentile) and Q3 (75th percentile)
        Q1 = self.df[column].quantile(0.25)
        Q3 = self.df[column].quantile(0.75)
        IQR = Q3 - Q1
            
            # Define outlier thresholds
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
            
        outliers = (self.df[column] < lower_bound) | (self.df[column] > upper_bound)
        outlier_count = outliers.sum()

        # Drop rows where the column has outliers
        self.df = self.df[~outliers]

        print(f"Removed {outlier_count} rows containing outliers in '{column}' using IQR method.")

    def remove_highly_correlated_columns(self, threshold=0.85):
        numeric_df = self.df.select_dtypes(include=['number'])  # Select only numeric columns
        corr_matrix = numeric_df.corr().abs()
        upper_triangle = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        to_drop = [column for column in upper_triangle.columns if any(upper_triangle[column] > threshold)]
        print(f"Removing highly correlated columns: {to_drop}")
        self.df.drop(columns=to_drop, inplace=True)


class Plotter:
    def __init__(self, df: pd.DataFrame):
        self.df = df

    def plot_boxplots(self):
        numeric_columns = self.df.select_dtypes(include=['number']).columns
        fig, axes = plt.subplots(4, 3, figsize=(15, 20)) 
        axes = axes.flatten() 

        for i, col in enumerate(numeric_columns):
            sns.boxplot(y=self.df[col], ax=axes[i])
            axes[i].set_title(col)
        plt.show()

    def plot_null_distribution(self):
        null_summary = self.df.isnull().mean() * 100
        null_summary.sort_values(ascending=False).plot(kind='bar', figsize=(10, 6))
        plt.title("NULL Value Distribution")
        plt.xlabel("Columns")
        plt.ylabel("Percentage of NULLs")
        plt.show()

    def plot_after_imputation(self, before_df: pd.DataFrame):
        before_null = before_df.isnull().sum()
        after_null = self.df.isnull().sum()

        comparison = pd.DataFrame({
            'Before Imputation': before_null,
            'After Imputation': after_null
        })

        comparison = comparison[comparison['Before Imputation'] > 0]

        # If there are no missing values in any column before imputation, skip plotting
        if comparison.empty:
            print("No missing values before imputation to plot.")
            return

        comparison.plot(kind='bar', figsize=(10, 6), title="Missing Values Before & After Imputation")
        plt.ylabel("Count of Missing Values")
        plt.xlabel("Columns")
        plt.xticks(rotation=45)
        plt.show()

    def plot_skewed_columns(self, skewed_columns):
        for column in skewed_columns:
            self.df[column].plot(kind='hist', figsize=(10, 6), bins=30)
            plt.title(f"Skewed Column: {column}")
            plt.xlabel(f"{column} Value")
            plt.ylabel("Frequency")
            plt.show()
            plt.close()

    def plot_transformed_columns(self):
        transformed_columns = [col for col in self.df.columns if '_log' in col or '_boxcox' in col or '_sqrt' in col]
        self.plot_skewed_columns(transformed_columns)

    def plot_correlation_matrix(self):
        numeric_df = self.df.select_dtypes(include=['number'])
        correlation_matrix = numeric_df.corr()

        plt.figure(figsize=(10, 8))
        sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
        plt.title("Correlation Matrix")
        plt.show()
###Testing the code

credentials = load_credentials('credentials.yaml')

connector = RDSDatabaseConnector(credentials)
connector.init_SQL_engine()

df = connector.fetch_data()

connector.save_to_csv(df, "failure_data.csv")

transformer = DataFrameTransform(df)
plotter = Plotter(df)

#Identifying and imputing nulls
before_imputation_df = df.copy()

null_summary = transformer.count_nulls()


    #Skew is close to 0 for all null columns, for Air temp and Process, null % is roughly 8.5
    #Tool wear is 4.84% so definitely impute
transformer.impute_nulls(strategy='mean')

# Identifying and rectifying skew
skewed_columns = transformer.identify_skewed_columns()


#plotter.plot_skewed_columns(skewed_columns)
transformer.reduce_skew(skewed_columns)
#plotter.plot_transformed_columns()

#transformer.save_transformed_data("transformed_data.csv")

#Identifying and removing outliers
transformer.remove_outliers(column='Rotational speed [rpm]', method='IQR')
transformer.remove_outliers(column='Torque [Nm]', method='IQR')

# Visualize again after removing outliers
#plotter = Plotter(transformer.df) 
#plotter.plot_boxplots()

#Visualise and remove overly correlated columns
plotter = Plotter(df)
plotter.plot_correlation_matrix()
#torque and rpm -0.92 correlation

transformer = DataFrameTransform(df)
transformer.remove_highly_correlated_columns()
