import streamlit as st
import pandas as pd
import numpy as np
import joblib
import xgboost as xgb
from sklearn.preprocessing import StandardScaler

# Load model and scaler
model = xgb.XGBClassifier()
model.load_model('model/xgboost_model.json')

scaler = joblib.load('model/feature_scaler.pkl')

# Load label encoders
team_encoder = joblib.load('model/team_encoder.pkl')
opponent_encoder = joblib.load('model/opponent_encoder.pkl')
venue_encoder = joblib.load('model/venue_encoder.pkl')
season_encoder = joblib.load('model/season_encoder.pkl')
round_encoder = joblib.load('model/round_encoder.pkl')  # Round encoder loaded

print(team_encoder.classes_)
print(opponent_encoder.classes_)
print(venue_encoder.classes_)
print(season_encoder.classes_)
print(round_encoder.classes_)


# Function to predict match result
def predict_match_result(team, opponent, venue, date, team_form, xg_pre_match, round):
    # Convert date to datetime
    date = pd.to_datetime(date)

    # Extract temporal features
    month = date.month
    day_of_week = date.dayofweek
    season = get_season(date)

    # Encode categorical variables
    team_encoded = team_encoder.transform([team])[0]
    opponent_encoded = opponent_encoder.transform([opponent])[0]
    venue_encoded = venue_encoder.transform([venue])[0]
    season_encoded = season_encoder.transform([season])[0]
    round_encoded = round_encoder.transform([round])[0]  # Encode the round using round_encoder

    # Prepare the feature vector for prediction
    features = np.array([[
        xg_pre_match, team_form, opponent_encoded, team_encoded, venue_encoded, 
        month, day_of_week, season_encoded, round_encoded
    ]])

    # Scale the features
    features_scaled = scaler.transform(features)

    # Predict using the model
    prediction = model.predict(features_scaled)
    
    # Map prediction to label
    result = ['Loss', 'Draw', 'Win'][prediction[0]]
    return result

# Helper function to get the season based on the date
def get_season(date):
    month = date.month
    if month in [12, 1, 2]:
        return 'winter'
    elif month in [3, 4, 5]:
        return 'spring'
    elif month in [6, 7, 8]:
        return 'summer'
    else:
        return 'autumn'

# Streamlit interface
st.title("Match Outcome Predictor")

# Possible teams, opponents, and venues (these can be dynamic based on the dataset)
teams = ['Arsenal', 'Aston Villa', 'Brentford', 'Brighton', 'Burnley', 'Chelsea',
 'Crystal Palace', 'Everton', 'Fulham', 'Leeds United', 'Leicester City',
 'Liverpool', 'Manchester City', 'Manchester Utd', 'Newcastle Utd',
 'Norwich City', 'Sheffield Utd', 'Southampton', 'Tottenham', 'Watford',
 'West Brom', 'West Ham', 'Wolves']
venues = ["Home", "Away"]
rounds = ['Matchweek 1', 'Matchweek 10', 'Matchweek 11', 'Matchweek 12', 'Matchweek 13',
 'Matchweek 14', 'Matchweek 15', 'Matchweek 16', 'Matchweek 17',
 'Matchweek 18', 'Matchweek 19', 'Matchweek 2', 'Matchweek 20', 'Matchweek 21',
 'Matchweek 22', 'Matchweek 23', 'Matchweek 24', 'Matchweek 25',
 'Matchweek 26', 'Matchweek 27', 'Matchweek 28', 'Matchweek 29', 'Matchweek 3',
 'Matchweek 30', 'Matchweek 31', 'Matchweek 32', 'Matchweek 33',
 'Matchweek 34', 'Matchweek 35', 'Matchweek 36', 'Matchweek 37',
 'Matchweek 38', 'Matchweek 4', 'Matchweek 5', 'Matchweek 6', 'Matchweek 7',
 'Matchweek 8', 'Matchweek 9']  # Tournament rounds

# User input
team = st.selectbox("Select Your Team", teams)
opponent = st.selectbox("Select Opponent", [t for t in teams if t != team])
# if team == opponent:
#     st.warning("You cannot select the same team for both your team and opponent!")
#     opponent = st.selectbox("Select Opponent", [t for t in teams if t != team])
venue = st.selectbox("Select Venue", venues)
date = st.date_input("Select Match Date")
team_form = st.slider("Team's Recent Form (Average Last 5 Matches)", 0.0, 1.0, 0.5)
xg_pre_match = st.number_input("Expected Goals (xG) Pre-Match", min_value=0.0, max_value=5.0, value=1.0)
round = st.selectbox("Tournament Round", rounds)  # Round selection input

# Prediction button
if st.button("Predict Result"):
    result = predict_match_result(team, opponent, venue, date, team_form, xg_pre_match, round)
    st.write(f"The predicted result is: **{result}**")
