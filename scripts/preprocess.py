import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder


def get_data(path):
    data = pd.read_csv(path)
    return data


def preprocess(data, train=False):
    fill_null_values(data)
    drop_null_values(data)
    data = add_feature_columns(data)
    drop_columns(data)
    data = encode_categorical_data(data)
    if train:
        X = data
        y = 0
    else:
        X = data.drop(columns=['Survived'], axis=1)
        y = data['Survived']

    return X, y


def drop_columns(data):
    data.drop(columns=['Name', 'PassengerId', 'Ticket', 'SibSp', 'Parch'
                       ], axis=1, inplace=True)


def drop_null_values(data):
    data.dropna(subset=['Embarked'], inplace=True)


def fill_null_values(data):
    data['Age'].fillna(int(data['Age'].mean()), inplace=True)
    data['Fare'].fillna(0, inplace=True)


def add_feature_columns(data):
    extract_family_size(data)
    refactor_deck_info(data)
    refactor_age(data)
    refactor_fare(data)
    extract_titles(data)
    return data


def refactor_age(data):
    data['Age_Bins'] = pd.qcut(data['Age'], 4, duplicates='drop')
    label = LabelEncoder()
    data['Age'] = label.fit_transform(data['Age_Bins'])
    data.drop(columns=['Age_Bins'], axis=1, inplace=True)


def refactor_fare(data):
    data['Fare_Bins'] = pd.qcut(data['Fare'], 4, duplicates='drop')
    label = LabelEncoder()
    data['Fare'] = label.fit_transform(data['Fare_Bins'])
    data.drop(columns=['Fare_Bins'], axis=1, inplace=True)


def refactor_deck_info(data):
    data['Cabin'].fillna('n')
    # get he deck label
    data['Deck'] = data['Cabin'].apply(lambda x: str(x)[0])
    # group labels
    data['Deck'] = data['Deck'].replace(['A', 'B', 'C', 'T'], 'ABCT')
    data['Deck'] = data['Deck'].replace(['D', 'E'], 'DE')
    data['Deck'] = data['Deck'].replace(['F', 'G'], 'FG')
    data['Deck'] = data['Deck'].replace(['n'], 'N')
    data.drop(columns=['Cabin'], axis=1, inplace=True)


def extract_family_size(data):
    data['Family_Size'] = data['SibSp']+data['Parch']+1
    data['Family_Size_Bins'] = pd.qcut(
        data['Family_Size'], 4, duplicates='drop')
    label = LabelEncoder()
    data['Family_Size'] = label.fit_transform(data['Family_Size_Bins'])
    data.drop(columns=['Family_Size_Bins'], axis=1, inplace=True)


def extract_titles(data):
    mapping = {'Mlle': 'Miss', 'Major': 'Mr', 'Col': 'Mr', 'Sir': 'Mr', 'Don': 'Mr', 'Mme': 'Miss',
               'Jonkheer': 'Mr', 'Lady': 'Mrs', 'Capt': 'Mr', 'Countess': 'Mrs', 'Ms': 'Miss', 'Dona': 'Mrs'}
    data['Title'] = data['Name'].apply(lambda x: extract_title(str(x)))
    data['Title'] = data['Title'].replace(mapping)


def encode_categorical_data(data):
    df_oh = pd.get_dummies(
        data=data,
        columns=["Sex", "Embarked", 'Deck', 'Title'
                 ],
        prefix=["Sex", "Embarked", 'Deck', 'Title'
                ])
    return df_oh


def extract_title(name):
    words = name.split()
    for word in words:
        if word.endswith('.'):
            return word[:-1]
