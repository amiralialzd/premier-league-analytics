import streamlit as st
import pandas as pd
import joblib

model = joblib.load('model.pkl')
df = pd.read_csv('premier_league_data.csv')


def get_team_features(hteam,ateam,df):
 df_a=df[(df['home_team']==ateam) |(df['away_team']==ateam)]
 df_h=df[(df['home_team']==hteam) |(df['away_team']==hteam)]
 df_a=df_a.sort_values(ascending=False,by='date')
 df_h=df_h.sort_values(ascending=False,by='date')
 first_row_h=df_h[0:1]
 first_row_a=df_a[0:1]
 full_team_features=[]
 if first_row_h['home_team'].values[0]==hteam:
     features=first_row_h[['home_team_total_wins','home_team_win_at_home' ,'home_team_win_away','home_team_last5_goals_scored','home_team_last5_goals_conceded']]
     features_h=features.to_dict('records')[0]
     full_team_features.append(features_h)
 elif first_row_h['away_team'].values[0]==hteam:
     features_h =  {
    'home_team_total_wins': first_row_h['away_team_total_wins'].values[0].item(),
    'home_team_win_at_home': first_row_h['away_team_win_at_home'].values[0].item(),
     'home_team_win_away' : first_row_h['away_team_win_away'].values[0].item(),
    'home_team_last5_goals_scored':first_row_h['away_team_last5_goals_scored'].values[0].item(),
    'home_team_last5_goals_conceded':first_row_h['away_team_last5_goals_conceded'].values[0].item()}
     full_team_features.append(features_h)

 if first_row_a['home_team'].values[0]==ateam:
     features_a={'away_team_total_wins':first_row_a['home_team_total_wins'].values[0].item(),
                 'away_team_win_at_home':first_row_a['home_team_win_at_home'].values[0].item(),
                 'away_team_win_away':first_row_a['home_team_win_away'].values[0].item(),
                 'away_team_last5_goals_scored':first_row_a['home_team_last5_goals_scored'].values[0].item(),
                 'away_team_last5_goals_conceded':first_row_a['home_team_last5_goals_conceded'].values[0].item()}
     full_team_features.append(features_a)

 elif first_row_a['away_team'].values[0]==ateam:
     features=first_row_a[['away_team_total_wins','away_team_win_at_home','away_team_win_away','away_team_last5_goals_scored','away_team_last5_goals_conceded']]
     features_a = features.to_dict('records')[0]
     full_team_features.append(features_a)

 full_dic=full_team_features[0] | full_team_features[1]
 return full_dic




st.title('Premier League Prediction ')


all_teams = sorted(df['home_team'].unique())
option_home=st.selectbox('choose home team ',all_teams)
option_away=st.selectbox('choose away team',all_teams)
if option_home == option_away:
    st.warning('Please select two different teams')
    st.stop()
features_col=['home_team_total_wins','away_team_total_wins','home_team_win_at_home','away_team_win_away','away_team_win_at_home',
              'home_team_win_away','home_team_last5_goals_scored','away_team_last5_goals_scored','home_team_last5_goals_conceded','away_team_last5_goals_conceded']
features=get_team_features(option_home,option_away,df)
input_row=pd.DataFrame([features])[features_col]
proba=model.predict_proba(input_row)

outcome_map = {0: 'Away Win', 1: 'Draw', 2: 'Home Win'}
prediction = model.predict(input_row)[0]
st.subheader(f'Predicted Result: {outcome_map[prediction]}')

proba_df=pd.DataFrame({'outcome':['Away Win','Draw','Home Win'],
                       'Probability': [proba[0][0], proba[0][1], proba[0][2]]})
st.bar_chart(proba_df.set_index('outcome'))
