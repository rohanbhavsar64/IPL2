import streamlit as st
import pandas as pd
import pickle
import plotly.graph_objects as go

# Load saved pipeline
with open("match_outcome_pipeline.pkl", "rb") as f:
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
city=st.text_input("city")
over = st.slider("Current Over", min_value=1, max_value=150, value=50)
wickets = st.slider("Wickets Lost", min_value=0, max_value=10, value=5)
current_score = st.number_input("Current Score", min_value=0, value=200)

first_inning_total = st.number_input("First Inning Total", min_value=0, value=300)
second_inning_total = st.number_input("Second Inning Total", min_value=0, value=200)
third_inning_total = st.number_input("Third Inning Total", min_value=0, value=250)
target = st.number_input("Target Score", min_value=0, value=350)

last_15_wickets = st.slider("Wickets Lost in Last 15 Overs", min_value=0, max_value=10, value=2)

if st.button("Predict Outcome"):
    input_df = pd.DataFrame([{
        "batting_team": batting_team,
        "bowling_team": bowling_team,
        "city":city,
        "over": over,
        "wickets": wickets,
        "current_score": current_score,
        "first_inning_total": first_inning_total,
        "second_inning_total": second_inning_total,
        "third_inning_total": third_inning_total,
        "target": target,
        "last_15_wickets": last_15_wickets
    }])

    probs = pipeline.predict_proba(input_df)[0]
    classes = pipeline.classes_
    fig = go.Figure(data=[go.Pie(
        labels=[batting_team,'draw',bowling_team],
        values=probs,
        hole=0.5,
        textinfo='label+percent',
        hoverinfo='label+value+percent',
        marker=dict(colors=['#1f77b4', '#ff7f0e', '#2ca02c'])
    )])

    fig.update_layout(
        title_text=f"📊 Prediction Probabilities",
        annotations=[dict(text='Predicator', x=0.5, y=0.5, font_size=20, showarrow=False)]
    )

    st.plotly_chart(fig)
