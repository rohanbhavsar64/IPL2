import pandas as pd
import numpy as np
import plotly.express as px
import streamlit as st
df=pd.read_csv('ipl.csv')
with open('iris_pipeline.pkl', 'rb') as f:
    pipe = pickle.load(f)
def match_progression(x_df,Id,pipe):
    match = x_df[x_df['Match ID'] ==Id]
    match = match[(match['balls_left']%6 == 0)]
    temp_df = match[['Match ID','Venue','Bat First','Bat Second','Batter','Non Striker','Bowler','Innings Runs','Innings Wickets','balls_left','crr','last_five']].fillna(0)
    temp_df = temp_df[temp_df['balls_left'] != 0]
    if temp_df.empty:
        print("Error: Match is not Existed")
        return None, None
    result = pipe.predict(temp_df)
    temp_df['projected_score'] = np.round(result,1)
    temp_df['end_of_over'] = range(1,temp_df.shape[0]+1)
    temp_df = temp_df[['end_of_over','Innings Runs','projected_score']]
    return temp_df

if option is not None:
    temp_df = match_progression(first,option,pipe)
    st.write(temp_df)
