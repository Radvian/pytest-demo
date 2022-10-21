import pandas as pd
import numpy as np

def postprocess(df):
    survive_decode = {1:'Survived', 0:'Did Not Survive'}
    df['Survived'] = df['Survived'].map(survive_decode)
    return df