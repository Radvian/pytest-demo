import pandas as pd
import numpy as np

def deck_column_removal(df):
    if 'Deck' in df.columns:
        df.drop('Deck', axis = 'columns', inplace = True)
        return df
    else:
        return df
    
def age_column_imputation(df):
    if df['Age'].isnull().sum() != 0:
        # Read imputation source
        imputation_source = pd.read_csv('C:/Users/User/pytest-demo/with_test/model_dev/age_imputation_df.csv')

        # Merge df with imputation source
        df = df.merge(imputation_source,  on = ['Sex', 'EmbarkedCity', 'Class'], how = 'left')

        # Change missing values in 'Age_x' with values from 'Age_y'
        df['Age'] = df['Age_x'].replace('',pd.NA).fillna(df['Age_y'])

        # Drop 'Age_x' and 'Age_y' columns
        df.drop(['Age_x', 'Age_y'], axis = 'columns', inplace = True)
    else:
        pass
    
    return df

def class_encoding(df):
    if df['Class'].dtype == 'O':
        classmap = {
            'First':1,
            'Second':2,
            'Third':3
        }
        df['Class'] = df['Class'].map(classmap)
    else:
        pass
    
    return df

def sex_encoding(df):
    if df['Sex'].dtype == 'O':
        gendermap = {'male':1, 'female':0}
        df['Sex'] = df['Sex'].map(gendermap)
    else:
        pass
    
    return df

def city_encoding(df):
    return pd.get_dummies(df)

def preprocess(df):
    df = city_encoding(sex_encoding(class_encoding(age_column_imputation(deck_column_removal(df)))))
    return df