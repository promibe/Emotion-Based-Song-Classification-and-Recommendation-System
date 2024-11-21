import pickle
import joblib
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from catboost import CatBoostClassifier
from feature_engineering import FeatureEngineering
# Load the pipeline
model = joblib.load('model_pipeline.pkl')

def predict_song_emotion(df):
    result = model.predict(df)
    # Extract the first value from the result (the prediction)
    if result is not None and isinstance(result, np.ndarray):
        result = result[0][0]
    return result

#print(predict_song_emotion(x))