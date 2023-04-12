import pandas as pd

def get_data(path):
    data = pd.read_csv(path)
    return data

def preprocess(data, train=False):
    data = add_feature_columns(data)
    drop_null_values(data)
    drop_columns(data)
    fill_null_values(data)
    data = encode_categorical_data(data)
    if train:
        X=data
        y=0
    else:
        X = data.drop(columns=['Survived'], axis=1)
        y = data['Survived']
        
    return X,y

def drop_columns(data):
    data.drop(columns=['Name', 'PassengerId', 'Ticket','Cabin'],axis=1,inplace=True)

def drop_null_values(data):
    data.dropna(subset=['Embarked'], inplace=True)
    

def fill_null_values(data):
    data['Age'].fillna(int(data['Age'].mean()), inplace=True)
    data['Fare'].fillna(0, inplace=True)

def add_feature_columns(data):
    #family size
    data['Family_Size'] = data['SibSp']+data['Parch']
    # cabin level
    data['Cabin'].fillna('n')
    # data['Cabin_Level']= data['Cabin'].apply(lambda x:str(x)[0])
    #family name
    # data['Family_Name'] = data['Name'].apply(lambda x: x.split(',')[0])
    return data

def encode_categorical_data(data):
    df_oh = pd.get_dummies(
    data=data,
    columns=["Sex", "Embarked"],
    prefix=["Sex", "Embarked"])
    return df_oh