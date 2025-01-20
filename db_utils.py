import yaml
import psycopg2
import pandas as pd
from sqlalchemy import create_engine

def load_credentials(file_path='credentials.yaml'):
    print("Loading credentials...")
    with open(file_path, 'r') as file:
        credentials = yaml.safe_load(file)
    print("Credentials loaded:", credentials) 
    return credentials


class RDSDatabaseConnector:
    def __init__ (self, credentials):
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
        data.to_csv(filename)
        print(f"Data successfully saved to {filename}")

def load_local_data(file_path='failure_data.csv'):
    return pd.read_csv(file_path)



credentials = load_credentials('credentials.yaml')

connector = RDSDatabaseConnector(credentials)
connector.init_SQL_engine()

df = connector.fetch_data()

connector.save_to_csv(df, "failure_data.csv")
print(f"Data shape: {df.shape}")
print(df.head())  