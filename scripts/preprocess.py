import pandas as pd

def get_data(path):
    data = pd.read_csv(path)
    return data

def drop_columns(data):
    return data.drop(columns=['Embarked'],axis=1)