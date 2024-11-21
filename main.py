import pickle
import joblib
from sklearn.pipeline import Pipeline
from catboost import CatBoostClassifier
from feature_engineering import FeatureEngineering
import streamlit as st      #importing streamlit
from model_file import predict_song_emotion
import numpy as np
import pandas as pd
import random

#creating the main function
def main():
    st.title("EMOTION-BASED MUSIC CLASSIFICATION AND RECOMMENDATION SYSTEM")

    html_template = """
    <div style="background-color:green;padding:10px">
    <h3 style="color:white;text-align:center;">leveraging Emotional Features for Better Music Recommendation</h3> 
    </div>
    """
    st.markdown(html_template, unsafe_allow_html=True)

    # Streamlit text inputs
    song_duration = st.text_input("duration (ms)", "Enter song time duration(min) -> 1.0 to 66.0")
    danceability = st.text_input("danceability", "Enter level (low: 0.0 - 0.3, normal: 0.31 - 0.6, high: 0.6 - 1.0)")
    energy = st.text_input("energy", "Enter level (low: 0.0 - 0.3, normal: 0.31 - 0.6, high: 0.6 - 1.0)")
    loudness = st.text_input("loudness",
                             "Enter level (v.low: -13.0 _ -10.0, low: -10.0 _ -7.0, normal: -7.0 _ -4.0, high: -4.0 _ -1.0, very high -1.0 _ 1)")
    speechiness = st.text_input("speechiness", "Enter level (low: 0.0 - 0.3, normal: 0.31 - 0.6, high: 0.6 - 1.0)")
    acousticness = st.text_input("acousticness", "Enter level (low: 0.0 - 0.3, normal: 0.31 - 0.6, high: 0.6 - 1.0)")
    instrumentalness = st.text_input("instrumentalness",
                                     "Enter level (low: 0.0 - 0.3, normal: 0.31 - 0.6, high: 0.6 - 1.0)")
    liveness = st.text_input("liveness", "Enter level (low: 0.0 - 0.3, normal: 0.31 - 0.6, high: 0.6 - 1.0)")
    valence = st.text_input("valence", "Enter level (low: 0.0 - 0.3, normal: 0.31 - 0.6, high: 0.6 - 1.0)")
    tempo = st.text_input("tempo","Enter level (slow: 0.0 - 56.25, moderate: 56.25 - 112.5, fast: 112.5 - 168.75, v.fast: 168.75 - 255)")
    spec_rate = st.text_input("spec_rate", "Enter level (slow: 0.0 ....... v.v.fast: 60)")

    result = ""

    if st.button("Predict"):
        try:
            # Convert all input values to float and validate ranges
            song_duration = float(song_duration)
            danceability = float(danceability)
            energy = float(energy)
            loudness = float(loudness)
            speechiness = float(speechiness)
            acousticness = float(acousticness)
            instrumentalness = float(instrumentalness)
            liveness = float(liveness)
            valence = float(valence)
            tempo = float(tempo)
            spec_rate = float(spec_rate)


            # Validate input ranges
            if not (1.0 <= song_duration <= 66.0):
                st.error("Song duration must be between 1.0 and 66.0 minutes.")
            elif not (0.0 <= danceability <= 1.0):
                st.error("Danceability must be between 0.0 and 1.0.")
            elif not (0.0 <= energy <= 1.0):
                st.error("Energy must be between 0.0 and 1.0.")
            elif not (-13.0 <= loudness <= 1.0):
                st.error("Loudness must be between -13.0 and 1.0.")
            elif not (0.0 <= speechiness <= 1.0):
                st.error("Speechiness must be between 0.0 and 1.0.")
            elif not (0.0 <= acousticness <= 1.0):
                st.error("Acousticness must be between 0.0 and 1.0.")
            elif not (0.0 <= instrumentalness <= 1.0):
                st.error("Instrumentalness must be between 0.0 and 1.0.")
            elif not (0.0 <= liveness <= 1.0):
                st.error("Liveness must be between 0.0 and 1.0.")
            elif not (0.0 <= valence <= 1.0):
                st.error("Valence must be between 0.0 and 1.0.")
            elif not (0.0 <= tempo <= 255.0):
                st.error("Tempo must be between 0.0 and 255.0.")
            elif not (0.0 <= spec_rate <= 60.0):
                st.error("Spec_rate must be between 0.0 and 60.0.")
            else:

                spec_rate = spec_rate / 1000000

                # If all inputs are valid, create the input data DataFrame
                input_data = pd.DataFrame({
                    'duration (ms)': [song_duration],
                    'danceability': [danceability],
                    'energy': [energy],
                    'loudness': [loudness],
                    'speechiness': [speechiness],
                    'acousticness': [acousticness],
                    'instrumentalness': [instrumentalness],
                    'liveness': [liveness],
                    'valence': [valence],
                    'tempo': [tempo],
                    'spec_rate': [spec_rate]
                })

                # Call the predict function (replace with your actual prediction logic)
                result = predict_song_emotion(input_data)
                print(f"Prediction result: {result}")  # Print to debug
                label = {0: 'sad song', 1: 'happy song', 2: 'energetic song', 3: 'calm song'}
                # Check if the result is a valid key
                if result in label:
                    st.success(f"User need Label {result}: {label[result]}")
                else:
                    st.error("Prediction result is invalid.")

        except ValueError:
            st.error("Please make sure all inputs are numeric and within the valid ranges.")

    if st.button('About:'):
        st.text(' I am Promise Ibediogwu Ekele - Data Scientist, ML/DL Engineer, - 07063083925')
        st.text(' Linkedin - Promise Ibediogwu')
        st.text('Built with Streamlit')



if __name__ == "__main__":
    main()


#"""
 #   'energy',
  #   'loudness',
   #  'speechiness',
    #'acousticness',
#     'instrumentalness',
 #    'liveness',
  #   'valence',
   #  'tempo',)
#    'spec_rate'
#"""