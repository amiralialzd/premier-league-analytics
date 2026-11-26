import pandas as pd
import numpy as np
import xgboost
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_sample_weight
import matplotlib.pyplot as plt
import joblib

df=pd.read_csv('ENG-Premier League 2024.csv')  # run from the repo folder
# df=df.drop(columns=['league','season','time','attendance','venue','referee','match_report','notes','game_id'])

results=[]
home_goals=[]
away_goals=[]

# def get_result():
#     for home,away,goals in zip(df['home_team'],df['away_team'],df['score']):
#         scores=goals.split('–')
#         h_score=int(scores[0])
#         a_score=int(scores[1])
#         home_goals.append(h_score)
#         away_goals.append(a_score)
#         if h_score>a_score:
#             results.append('H')
#         elif h_score<a_score :
#             results.append('A')
#
#         else:
#             results.append('D')






# get_result()

# df['result']=results
# df['home_goals']=home_goals
# df['away_goals']=away_goals
# print(df[['home_team', 'away_team','result', 'score','home_goals','away_goals']].head(10))
# df=df.drop(columns='score',axis=1)
df['matches']=df['home_team'] +'vs'+df['away_team']


rest={}
home_last5=[]
away_last5=[]
for name,res  in zip(df['matches'],df['result']):
    teams=name.split('vs')
    a_team=teams[1]
    h_team=teams[0]
    rest.setdefault(h_team, [])
    rest.setdefault(a_team, [])
    home_last5.append(rest[h_team][-5:])
    away_last5.append(rest[a_team][-5:])

    if res=='H':
        rest[h_team].append('W')
        rest[a_team].append('L')
    elif res=='D':
        rest[h_team].append('D')
        rest[a_team].append('D')
    else:
        rest[h_team].append('L')
        rest[a_team].append('W')

df['home_last5'] = home_last5
df['away_last5'] = away_last5
# print(df.tail())

home_avg=[]
for i in df['home_last5']:
   home_tot =0
   avg=0
   for j in i:
      if j=='W':
          home_tot+=1
      elif j=='D':
          home_tot+=0.5
   if len(i)!=0:
    avg=home_tot/len(i)
    home_avg.append(avg)
   else:
       home_avg.append(0)

away_avg=[]
for i in df['away_last5']:
   away_tot = 0
   avg=0
   for j in i:
      if j=='W':
          away_tot+=1
      elif j=='D':
          away_tot+=0.5
   if len(i)!=0:
       avg=away_tot/len(i)
       away_avg.append(avg)
   else:
       away_avg.append(0)

df['away_form']=away_avg
df['home_form']=home_avg
teams={}
win_collector=[]
win_collector_a=[]
for team, result,team_a in zip(df['home_team'],df['result'],df['away_team']):
    teams.setdefault(team,0)
    teams.setdefault(team_a,0)
    win_collector.append(teams[team])
    win_collector_a.append(teams[team_a])
    if result=='H':
      teams[team]+=1
    elif result=='A':
        teams[team_a]+=1

df['home_team_total_wins']=win_collector
df['away_team_total_wins']=win_collector_a


home_team_win_at_home={}
home_team_win_collector=[]
for team,result in zip(df['home_team'],df['result']):
    home_team_win_at_home.setdefault(team,0)
    home_team_win_collector.append(home_team_win_at_home[team])
    if result=='H':
        home_team_win_at_home[team]+=1


df['home_team_win_at_home']=home_team_win_collector

away_win_at_away={}
away_team_win_collector=[]
for team , result in zip(df['away_team'],df['result']):
    away_win_at_away.setdefault(team,0)
    away_team_win_collector.append(away_win_at_away[team])
    if result=='A':
        away_win_at_away[team]+=1


df['away_team_win_away']=away_team_win_collector


pd.set_option('display.max_columns', None)
# print(df[['home_team','away_team', 'result', 'away_team_win_away', 'home_team_win_at_home', 'away_team_total_wins','home_team_total_wins']].iloc[20:70])

away_team_win_home=[]
home_team_win_away=[]
away_teams={}
home_teams={}
for team_h, result, team_a in zip(df['home_team'],df['result'],df['away_team']):
    home_teams.setdefault(team_a,0)
    away_teams.setdefault(team_h,0)
    home_teams.setdefault(team_h,0)
    away_teams.setdefault(team_a,0)
    away_team_win_home.append(home_teams[team_a])
    home_team_win_away.append(away_teams[team_h])
    if result=='H':
        home_teams[team_h] += 1
    elif result=='A':
        away_teams[team_a]+=1


df['away_team_win_at_home']=away_team_win_home
df['home_team_win_away']=home_team_win_away

# print(df.info())

# jun9=df.to_csv('jun9 update for premier league project.csv')
home_goal_scored=[]
home_goals_teams={}
for goals, team in zip(df['home_goals'],df['home_team']):
    home_goals_teams.setdefault(team,[])
    home_goal_scored.append(home_goals_teams[team][-5:])
    home_goals_teams[team].append(goals)

avg_last5_goal_scored=[]
for list in home_goal_scored:
    tot=0
    for j in list:
        tot+=j
    if len(list)==0:
        avg_last5_goal_scored.append(0)

    else:
        avg = tot / len(list)
        avg_last5_goal_scored.append(avg)

df['home_team_last5_goals_scored']=avg_last5_goal_scored
away_goal_scored=[]
away_goals_teams={}
for goals,team in zip(df['away_goals'],df['away_team']):
    away_goals_teams.setdefault(team,[])
    away_goal_scored.append(away_goals_teams[team][-5:])
    away_goals_teams[team].append(goals)

avg_away_last5=[]
for list in away_goal_scored:
    tot=0
    for j in list:
        tot+=j
    if len(list)==0:
        avg_away_last5.append(0)
    else:
        avg=tot/len(list)

        avg_away_last5.append(avg)

df['away_team_last5_goals_scored']=avg_away_last5

home_goals_conceded=[]
home_goals_team={}
for team, goals in zip(df['home_team'],df['away_goals']):
    home_goals_team.setdefault(team,[])
    home_goals_conceded.append(home_goals_team[team][-5:])
    home_goals_team[team].append(goals)

avg_home_team_goal_conceded=[]
for list in home_goals_conceded:
    tot=0
    for j in list:
        tot+=j
    if len(list)==0:
        avg_home_team_goal_conceded.append(0)
    else:
        avg=tot/len(list)
        avg_home_team_goal_conceded.append(avg)
df['home_team_last5_goals_conceded']=avg_home_team_goal_conceded


away_goals_conceded=[]
away_goals_team={}
for team, goals in zip(df['away_team'],df['home_goals']):
    away_goals_team.setdefault(team,[])
    away_goals_conceded.append(away_goals_team[team][-5:])
    away_goals_team[team].append(goals)

avg_away_team_goal_conceded=[]
for list in away_goals_conceded:
    tot=0
    for j in list:
        tot+=j
    if len(list)==0:
        avg_away_team_goal_conceded.append(0)
    else:
        avg=tot/len(list)
        avg_away_team_goal_conceded.append(avg)

df['away_team_last5_goals_conceded']=avg_away_team_goal_conceded


df=df.drop(columns=['Unnamed: 0','home_goals','away_goals','matches'])
df=df.drop(columns=['home_last5','away_last5','day','week','game'])

X=df.drop(columns=['home_team','away_team','result','home_form','away_form','date'])
y=df['result']

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,stratify=y,random_state=42)
le = LabelEncoder()
y_train = le.fit_transform(y_train)
y_test = le.transform(y_test)



scaler=StandardScaler()
regression=LogisticRegression(max_iter=100)
reg_pip=Pipeline(steps=[('scaler',scaler),('logistic',regression)])
reg_pip.fit(X_train,y_train)
y_pred=reg_pip.predict(X_test)
accuracy=accuracy_score( y_test,y_pred)
print(f'accuracy for logistic regression: {accuracy}')
print(classification_report(y_test, y_pred))



randomForest=RandomForestClassifier(random_state=42)
random_pip=Pipeline(steps=[('scaler',scaler),('logestic',randomForest)])
random_pip.fit(X_train,y_train)
y_pred_random=random_pip.predict(X_test)
forest_acc=accuracy_score(y_test,y_pred_random)
print(f'random forest accuracy score  : {forest_acc}')
print(classification_report(y_test, y_pred_random))

xg=xgboost.XGBClassifier(random_state=42)
xg_pip=Pipeline(steps=[('scaler',scaler),('logistic',xg)])
xg_pip.fit(X_train,y_train)
y_pred_xg=xg_pip.predict(X_test)
print(f'classification report for xg: {classification_report(y_test,y_pred_xg)}')
sample_weights = compute_sample_weight(class_weight='balanced', y=y_train)
param={'logistic__n_estimators':[100, 200, 300, 500],
       'logistic__max_depth':[3, 5, 7, 10],
       'logistic__subsample':[0.6, 0.7, 0.8, 0.9],
       'logistic__learning_rate':[0.01, 0.05, 0.1, 0.2]
       }
grid=GridSearchCV(param_grid=param,estimator=xg_pip,cv=5,scoring='f1_macro')
grid.fit(X_train,y_train,logistic__sample_weight=sample_weights)
score=grid.best_score_
best=grid.best_estimator_
print(f'best estimator {best} and score {score}  ')

y_pred_tuned = grid.best_estimator_.predict(X_test)
print(classification_report(y_test, y_pred_tuned))




model = grid.best_estimator_.named_steps['logistic']
features = X.columns
importance = model.feature_importances_

feat_imp = pd.Series(importance, index=features).sort_values(ascending=False)
feat_imp.plot(kind='bar')
plt.title('Feature Importance')
plt.tight_layout()
# plt.show()
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


features=get_team_features('Arsenal', 'Manchester City', df)
input_row=pd.DataFrame([features])[X.columns]
prediction=grid.best_estimator_.predict(input_row)
proba = grid.best_estimator_.predict_proba(input_row)
print(X.columns.tolist())
joblib.dump(grid.best_estimator_, 'model.pkl')
model = joblib.load('model.pkl')
df.to_csv('premier_league_data.csv', index=False)