import pandas as pd
import numpy as np
import plotly.express as px
from matplotlib import pyplot as plt
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
print(df)
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
print(first[first['Match ID']==1359507])
x=first.drop(columns='Target Score')
y=first['Target Score']
from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest= train_test_split(x,y,test_size=0.2,random_state=1)
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ohe=OneHotEncoder()
trf = ColumnTransformer([
    ('trf',OneHotEncoder(sparse_output=False,handle_unknown = 'ignore'),['Venue','Bat First','Bat Second','Batter','Non Striker','Bowler'])],remainder='passthrough')
from sklearn.linear_model import LinearRegression
from sklearn import linear_model
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
pipe=Pipeline(
    steps=[
        ('step1',trf),
        ('step2',LinearRegression())
    ])
pipe.fit(xtrain,ytrain)
print(pipe.predict(xtest)[1])

#n=pipe.predict_proba(pd.DataFrame(columns=['batting_team','bowling_team','city','batsman','non_striker','runs_left','ball_left','wickets_left','total_runs_x','crr','rrr','last_five','last_five_wicket'],data=np.array(['Royal Challengers Bangalore','Chennai Super Kings','Indore','Dhoni','Sundar',63,42,7,2,11.23,9.00,33,2.0]).reshape(1,13))).astype(float)

#print("Win Chances of Batting team is:", n[0][1]*100,"%")
#print("Win Chances of Bowling team is:", n[0][0]*100,"%")



