import pandas as pd
from bs4 import BeautifulSoup
import requests
import streamlit as st
# --- INPUT ---
url = 'https://www.espncricinfo.com/series/icc-cricket-world-cup-2023-24-1367856/australia-vs-south-africa-2nd-semi-final-1384438/match-overs-comparison'
max_over = 50  # Adjust as needed

# --- FETCH & PARSE ---
response = requests.get(url)
soup = BeautifulSoup(response.text, 'html.parser')
elements = soup.find_all(class_='ds-cursor-pointer ds-pt-1')

# --- DATA EXTRACTION ---
first_innings_runs, first_innings_wickets, overs_1 = [], [], []
second_innings_runs, second_innings_wickets, overs_2 = [], [], []

for i, element in enumerate(elements):
    if '/' not in element.text:
        continue
    run, wicket = element.text.split('/')
    wicket = wicket.split('(')[0]
    run = int(run.strip())
    wicket = int(wicket.strip())

    if i % 2 == 0:
        first_innings_runs.append(run)
        first_innings_wickets.append(wicket)
        overs_1.append(i/2+1)
    else:
        second_innings_runs.append(run)
        second_innings_wickets.append(wicket)
        overs_2.append(i/2+1)

# Limit by selected over
first_innings_runs = first_innings_runs[:max_over]
first_innings_wickets = first_innings_wickets[:max_over]
overs_1 = overs_1[:max_over]

second_innings_runs = second_innings_runs[:max_over]
second_innings_wickets = second_innings_wickets[:max_over]
overs_2 = overs_2[:max_over]

# --- DATAFRAMES ---
df1 = pd.DataFrame({'over': overs_1, 'score': first_innings_runs, 'wickets': first_innings_wickets})
df2 = pd.DataFrame({'over': overs_2, 'score': second_innings_runs, 'wickets': second_innings_wickets})

# --- PLOTTING ---
plt.figure(figsize=(10, 5))
plt.plot(df1['over'], df1['score'], label='1st Innings', color='red', linewidth=2, marker='o')
plt.plot(df2['over'], df2['score'], label='2nd Innings', color='green', linewidth=2, marker='o')

# Wicket markers
for i in range(1, len(df1)):
    if df1['wickets'].iloc[i] > df1['wickets'].iloc[i - 1]:
        plt.scatter(df1['over'].iloc[i], df1['score'].iloc[i], color='red', marker='x', s=100)

for i in range(1, len(df2)):
    if df2['wickets'].iloc[i] > df2['wickets'].iloc[i - 1]:
        plt.scatter(df2['over'].iloc[i], df2['score'].iloc[i], color='green', marker='x', s=100)

st.write(df1)
st.write(df2)
