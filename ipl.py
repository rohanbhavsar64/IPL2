import pandas as pd
import numpy as np
import plotly.express as px
import streamlit as st
import pickle
df=pd.read_csv('ipl.csv')
df=pd.read_csv('ball_by_ball_ipl.csv')
df['Over'] = df['Over'].astype(int)
df['Ball'] = df['Ball'].astype(int)
import pandas as pd

# Assuming Over and Ball columns are integers
df['Balls'] = df['Over'] * 6 + df['Ball']

# Sort the data to ensure correct order
df = df.sort_values(by=['Match ID', 'Innings', 'Balls'])

# Group by match and innings to calculate rolling difference
def compute_last_five(group):
    group = group.copy()
    group['last_five'] = group['Innings Runs'] - group['Innings Runs'].shift(18)
    group['last_five'] = group['last_five'].fillna(0)  # For first 18 balls, set to 0
    return group

df = df.groupby(['Match ID', 'Innings']).apply(compute_last_five).reset_index(drop=True)
#print(df)
df['year'] = pd.to_datetime(df['Date']).dt.year
# Group by Batter and sum the 'Innings Runs' (or use the correct runs column)
batter_runs = df.groupby('Batter')['Batter Runs'].sum().reset_index()
top_batters = batter_runs.sort_values(by='Batter Runs', ascending=False).head(10)
fig = px.bar(top_batters, x='Batter', y='Batter Runs', title='Top 10 Batters by Total Runs')
# Filter for only the first innings
first_innings = df[df['Innings'] == 1]

# Group by Match ID and Year to get total first innings score per match
match_scores = first_innings.groupby(['Match ID', 'year'])['Innings Runs'].max().reset_index()

# Now group by year to get average first innings score
yearly_avg = match_scores.groupby('year')['Innings Runs'].mean().reset_index()
# Plotting with Matplotlib
fig = px.bar(yearly_avg, x='year', y='Innings Runs', title='Avg. 1st Innings Scores per year')
df1=df[['Match ID','Venue','Bat First','Bat Second','Innings','Over','Ball','Batter','Non Striker','Bowler','Innings Runs','Innings Wickets','Target Score','Runs to Get','Balls Remaining','last_five','Winner']]
df1['balls_left']=120-(6*(df1['Over']-1)+df1['Ball'])

first=df1[df1['Innings']==1]
first['crr']=first['Innings Runs']*6/(120-first['balls_left'])

first=first[['Match ID','Venue','Bat First','Bat Second','Batter','Non Striker','Bowler','Innings Runs','Innings Wickets','balls_left','crr','last_five','Target Score']]
#print(first[first['Match ID']==1359507])
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
option=st.selectbox('Enter match ID',first['Match ID'].unique())
if option is not None:
    temp_df = match_progression(first,option,pipe)
    st.write(temp_df)
