import pandas as pd
import numpy as np
import pytest
import os
import sys

sys.path.append('..')

from src.preprocess import *

def test_deck_column_removal_withdeck():
    dummy_df = pd.DataFrame({'Col_1':[0,1], 'Deck':[1,2]})
    assert deck_column_removal(dummy_df).shape[1] == 1
    
def test_deck_column_removal_withoutdeck():
    dummy_df = pd.DataFrame({'Col_1':[0,1], 'NotDeck':[1,2]})
    assert deck_column_removal(dummy_df).shape[1] == 2
    
def test_age_column_imputation():
    dummy_df = pd.DataFrame({'Age':[np.nan, 20,np.nan],
                             'Sex':['male', 'female', 'female'],
                             'EmbarkedCity':['S','Q','C'],
                             'Class':['First','Second','Third']})
    age_imputed_df = age_column_imputation(dummy_df)
    assert age_imputed_df['Age'].values[0] == 42.00
    assert age_imputed_df['Age'].values[1] == 20.00
    assert age_imputed_df['Age'].values[2] == 14.25
    
def test_class_encoding():
    dummy_df = pd.DataFrame({'Class':['First', 'Second', 'Third']})
    assert class_encoding(dummy_df)['Class'].values[0] == 1
    assert class_encoding(dummy_df)['Class'].values[1] == 2
    assert class_encoding(dummy_df)['Class'].values[2] == 3
    
def test_sex_encoding():
    dummy_df = pd.DataFrame({'Sex':['male', 'female']})
    assert sex_encoding(dummy_df)['Sex'].values[0] == 1
    assert sex_encoding(dummy_df)['Sex'].values[1] == 0
    
def test_city_encoding():
    dummy_df = pd.DataFrame({'City':['A','B','C']})
    encoded_df = city_encoding(dummy_df)
    assert (encoded_df.columns == ['City_A', 'City_B', 'City_C']).sum() == 3
    assert encoded_df.sum(axis = 1).sum() == 3
    
def test_preprocess():
    dummy_df = pd.DataFrame({'Deck':['A','B'],
                             'Age':[np.nan,50],
                             'Sex':['male', 'female'],
                             'EmbarkedCity':['S','Q'],
                             'Class':['Second','First']})
    preprocessed_df = preprocess(dummy_df)
    
    assert (preprocessed_df['Age'].values == [30, 50]).sum() == 2
    assert preprocessed_df.shape[1] == 5
    
    
#### Testing with Fixtures

@pytest.fixture
def load_dummy_df():
    dummy_df = pd.DataFrame({'Deck':['A','B'],
                             'Age':[np.nan,50],
                             'Sex':['male', 'female'],
                             'EmbarkedCity':['S','Q'],
                             'Class':['Second','First']})
    return dummy_df

def test_deck_removal_withfixture(load_dummy_df):
    updated_df = deck_column_removal(load_dummy_df)
    assert 'Deck' not in updated_df.columns
    
def test_age_imputation_withfixture(load_dummy_df):
    assert (age_column_imputation(load_dummy_df)['Age'].values == [30,50]).sum() == 2    