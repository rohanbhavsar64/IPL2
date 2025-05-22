import streamlit as st
import pandas as pd
import pickle
import zipfile
import os

# --- Extract ZIP if not already done ---
zip_file = "match_outcome_pipeline1.zip"
extracted_model_path = "match_outcome_pipeline.pkl"

if not os.path.exists(extracted_model_path):
    with zipfile.ZipFile(zip_file, 'r') as zip_ref:
        zip_ref.extractall()

# --- Load the pipeline ---
with open(extracted_model_path, "rb") as f:
    pipeline = pickle.load(f)

st.set_page_config(page_title="Cricket Match Outcome Predictor", layout="centered")
st.title("🏏 Cricket Match Outcome Predictor")

st.markdown("Enter match details below to predict the **outcome probabilities**:")

# Team options (should match training data)
team_options = [
    'India', 'Australia', 'England', 'Pakistan', 'New Zealand',
    'South Africa', 'Sri Lanka', 'Bangladesh', 'Afghanistan', 'West Indies'
]

# Input widgets
batting_team = st.selectbox("Batting Team", team_options)
bowling_team = st.selectbox("Bowling Team", team_options)

over = st.slider("Current Over", min_value=1, max_value=150, value=50)
wickets = st.slider("Wickets Lost", min_value=0, max_value=10, value=5)
current_score = st.number_input("Current Score", min_value=0, value=200)

first_inning_total = st.number_input("First Inning Total", min_value=0, value=300)
second_inning_total = st.number_input("Second Inning Total", min_value=0, value=200)
third_inning_total = st.number_input("Third Inning Total", min_value=0, value=250)
target = st.number_input("Target Score", min_value=0, value=350)

last_15_wickets = st.slider("Wickets Lost in Last 15 Overs", min_value=0, max_value=10, value=2)

if st.button("Predict Outcome"):
    # Prepare input
    input_data = pd.DataFrame([{
        "batting_team": batting_team,
        "bowling_team": bowling_team,
        "over": over,
        "wickets": wickets,
        "current_score": current_score,
        "first_inning_total": first_inning_total,
        "second_inning_total": second_inning_total,
        "third_inning_total": third_inning_total,
        "target": target,
        "last_15_wickets": last_15_wickets
    }])

    # Prediction
    probs = pipeline.predict_proba(input_data)[0]
    labels = pipeline.classes_

    st.subheader("🧠 Predicted Outcome Probabilities:")
    st.write(f"Batting: **{batting_team}** vs Bowling: **{bowling_team}**")

    for label, prob in zip(labels, probs):
        st.write(f"**{label.upper()}**: {prob*100:.2f}%")

    st.success("Prediction complete!")
