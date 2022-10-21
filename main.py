import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import pickle
import sys
import os
import logging


from src.preprocess import preprocess
from src.postprocess import postprocess

def main():
    logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p', level = logging.INFO)
    
    
    # Load model
    model = pickle.load(open('model_dev/rfc_model.sav', 'rb'))
    logging.info('Model loaded!')
    
    # Load to_predict data
    data = pd.read_csv('prediction_data/to_predict.csv')
    logging.info('To-Predict data loaded!')
    
    # Preprocess to_predict data
    clean_data = preprocess(data)
    logging.info('To-Predict data cleaned!')
    
    # Make Prediction
    logging.info('Making prediction...')
    prediction = model.predict(clean_data)
    clean_data['Survived'] = prediction
    logging.info('Done making prediction!')
    
    # Postprocessing
    clean_data = postprocess(clean_data)
    
    # Saving Prediction
    clean_data.to_csv('prediction_result.csv', index = False)
    logging.info('Prediction result saved! Finished.')
    
if __name__ == "__main__":
    print("Start!")
    main()