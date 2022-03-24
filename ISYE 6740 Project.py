# -*- coding: utf-8 -*-
"""
Created on Fri Jul 16 11:05:29 2021

@author: StoneHayden
"""

import pandas as pd
import os
import numpy as np
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import plotly.express as px
from sklearn.decomposition import PCA
path = os.getcwd() + '\\pbp-2015.csv'    #get path

df = pd.read_csv(path)
df = df[~(df['Quarter'] == 5)] #drop OT quarters due to different rules
df = df[~(df['Down'] == 0)] 

df['TimeRemaining'] = ((4 - df['Quarter'])*900) + (df['Minute']*60) + df['Second'] # converter Quarter/Minute/Seconds columns into Time Remaining in seconds
df['YardsToEndzone'] = 100 - df['YardLine'] #figure out how far from end zone we are

df['IsShotgun'] = df["Formation"].str.contains("SHOTGUN") * 1       #get formation - ie whether shotgun or under center, and whether is no huddle
df['IsUnderCenter'] = df["Formation"].str.contains("SHOTGUN")==False * 1
df['IsNoHuddle'] = df["Formation"].str.contains("HUDDLE") * 1

num_allplays = df.shape[0]      #number of plays in 2015

playtype = ['PASS', 'RUSH', 'SACK', 'SCRAMBLE']     #FILTER BY PLAYTYPE

df = df[df['PlayType'].isin(playtype)]
df['PlayType'].loc[df['PlayType'].str.contains('SACK')] = 'PASS'
df['PlayType'].loc[df['PlayType'].str.contains('SCRAMBLE')] = 'PASS'

df['IsPass'] = (df["PlayType"] == 'PASS') * 1     #1 is pass, 0 is rush

nfl_teams = df.OffenseTeam.unique().tolist()

num_wantedplays = df.shape[0]       #number of pass/rush plays
print("Passes and Rushes make up ", num_wantedplays / num_allplays, " of all NFL plays in 2015.")
print("\n")


final_df =df[['TimeRemaining', 'YardsToEndzone', 'OffenseTeam','Down','ToGo','SeriesFirstDown', 'IsShotgun', 'IsUnderCenter', 'IsNoHuddle', 'PassType', 'RushDirection', 'PlayType', 'IsPass']] #Parse down df to only wanted columns

# all_y = final_df['IsPass']
# all_x = final_df.drop(['OffenseTeam', 'IsPass', 'PlayType', 'PassType', 'RushDirection'], axis = 1)
# all_xtrain, all_xtest, all_ytrain, all_ytest = train_test_split(all_x,all_y, test_size=0.2)

# #KNN Test
# all_scores = []
# for i in range(20):
#     knn = KNeighborsClassifier(n_neighbors=1+i)
#     knn.fit(all_xtrain, all_ytrain)
#     all_scores.append(knn.score(all_xtest, all_ytest))
# max_knn_allplays = max(all_scores)


# #SVM Test
# svm = SVC(kernel="linear", C=0.025)
# svm.fit(all_xtrain, all_ytrain)
# all_SVM = svm.score(all_xtest, all_ytest)

# #RF Test
# forest = RandomForestClassifier(max_depth=5)
# forest.fit(all_xtrain, all_ytrain)
# all_forest = forest.score(all_xtest, all_ytest)


# sns.catplot(x='PlayType', kind='count', data=final_df, orient='h')
# plt.title('2015 All Games: Pass vs Run Counts')
# plt.xlabel("Pass vs Run")
# plt.ylabel("Count")
# plt.show()

# sns.catplot(x="Down", kind="count", hue='PlayType', data=final_df);
# plt.title('2015 All Games: Pass/Run Splits per Down')
# plt.xlabel("Pass vs Run per Down")
# plt.ylabel("Count")
# plt.show()

# sns.lmplot(x="ToGo", y="IsPass", data=final_df, y_jitter=.03, logistic=True, aspect=2)
# plt.title('2015 All Games: Yards To Go vs Pass Likelihood')
# plt.xlabel("Yards To Go")
# plt.ylabel("Pass Likelihood")
# plt.show()

# sns.lmplot(x="TimeRemaining", y="IsPass", data=final_df, y_jitter=.03, logistic=True, aspect=2)
# plt.title('2015 All Games: Time Remaining vs Pass Likelihood')
# plt.xlabel("Time Remaining")
# plt.ylabel("Pass Likelihood")
# plt.show()


# car_df = final_df[final_df['OffenseTeam'] == 'CAR']


# sns.catplot(x='PlayType', kind='count', data=car_df, orient='h')
# plt.title('2015 Panthers: Pass vs Run Counts')
# plt.xlabel("Pass vs Run")
# plt.ylabel("Count")
# plt.show()

# sns.catplot(x="Down", kind="count", hue='PlayType', data=car_df);
# plt.title('2015 Panthers: Pass/Run Splits per Down')
# plt.xlabel("Pass vs Run per Down")
# plt.ylabel("Count")
# plt.show()

# sns.catplot(x="ToGo", kind="count", hue='PlayType', data=car_df);
# plt.title('2015 Panthers: Pass/Run Splits Based on Yards To Go')
# plt.xlabel("Pass/Run per Yards To Go")
# plt.ylabel("Count")
# plt.show()

# sns.lmplot(x="ToGo", y="IsPass", data=car_df, y_jitter=.03, logistic=True, aspect=2)
# plt.title('2015 Panthers: Yards To Go vs Pass Likelihood')
# plt.xlabel("Yards To Go")
# plt.ylabel("Pass Likelihood")
# plt.show()

# sns.lmplot(x="TimeRemaining", y="IsPass", data=car_df, y_jitter=.03, logistic=True, aspect=2)
# plt.title('2015 Panthers: Time Remaining vs Pass Likelihood')
# plt.xlabel("Time Remaining")
# plt.ylabel("Pass Likelihood")
# plt.show()

# y = car_df['IsPass']
# x = car_df.drop(['OffenseTeam', 'IsPass', 'PlayType', 'PassType', 'RushDirection'], axis = 1)

# xtrain, xtest, ytrain, ytest = train_test_split(x,y, test_size=0.2)

# #KNN Test
# scores = []
# for i in range(20):
#     knn = KNeighborsClassifier(n_neighbors=1+i)
#     knn.fit(xtrain, ytrain)
#     scores.append(knn.score(xtest, ytest))    
# max_car_knn = max(scores)


# print("For testing 1-20 knn neighbors, the max accuracy score we achieved on all 2015 NFL plays is: ", max_knn_allplays)
# print("For testing 1-20 knn neighbors, the max accuracy score we achieved on the 2015 Panthers is: ", max_car_knn)
# print("\n")

# #SVM Test
# svm = SVC(kernel="linear", C=0.025)
# svm.fit(xtrain, ytrain)
# panthers_svm = svm.score(xtest, ytest)
# print("Accuracy Score of SVM on All 2015 NFL Plays is: ", all_SVM)
# print("Accuracy Score of SVM on 2015 Panthers Plays is: ", panthers_svm)
# print("\n")

# #RF Test
# forest = RandomForestClassifier(max_depth=5)
# forest.fit(xtrain, ytrain)
# panthers_forest = forest.score(xtest, ytest)
# print("Accuracy Score of Random Forest on all 2015 NFL Plays is: ", all_forest)
# print("Accuracy Score of Random Forest on 2015 Panthers Plays is: ", panthers_forest)
# print("\n")

# import matplotlib.pyplot as plt
# import pandas as pd
# from pandas.plotting import table
# ax = plt.subplot(111, frame_on=False) # no visible frame
# ax.xaxis.set_visible(False)  # hide the x axis
# ax.yaxis.set_visible(False)  # hide the y axis
# table(ax, car_df)  # where df is your data frame
# plt.show()

nfl_scores = pd.DataFrame(nfl_teams, columns = ['Team'])
nfl_scores = nfl_scores.set_index('Team')
nfl_scores['KNN score'] = 0
nfl_scores['SVM score'] = 0
nfl_scores['RF score'] = 0

for team in nfl_teams:
    team_df = final_df[final_df['OffenseTeam'] == team]
    y = team_df['IsPass']
    x = team_df.drop(['OffenseTeam', 'IsPass', 'PlayType', 'PassType', 'RushDirection'], axis = 1)
    
    pca = PCA(n_components = 5)
    pca.fit(x)
    x = pca.transform(x)
    xtrain, xtest, ytrain, ytest = train_test_split(x,y, test_size=0.2)   
    knn = KNeighborsClassifier(n_neighbors=10)
    knn.fit(xtrain, ytrain)
    knn_score = knn.score(xtest, ytest)       
    
    #SVM Test
    svm = SVC(kernel="linear", C=0.025)
    svm.fit(xtrain, ytrain)
    svm_score = svm.score(xtest, ytest)

    #RF Test
    forest = RandomForestClassifier(max_depth=5)
    forest.fit(xtrain, ytrain)
    forest_score = forest.score(xtest, ytest)

    nfl_scores.loc[team] = [knn_score, svm_score, forest_score]

nfl_scores['Total Accuracy score'] =nfl_scores['KNN score'] * nfl_scores['SVM score'] * nfl_scores['RF score']
nfl_scores['Wins'] = 0
nfl_scores['Wins'].loc['CAR'] = 15
nfl_scores['Wins'].loc['ARI'] = 13
nfl_scores['Wins'].loc['CIN'] = 12
nfl_scores['Wins'].loc['DEN'] = 12
nfl_scores['Wins'].loc['NE'] = 12
nfl_scores['Wins'].loc['KC'] = 11
nfl_scores['Wins'].loc['MIN'] = 11
nfl_scores['Wins'].loc['GB'] = 10
nfl_scores['Wins'].loc['NYJ'] = 10
nfl_scores['Wins'].loc['PIT'] = 10
nfl_scores['Wins'].loc['SEA'] = 10
nfl_scores['Wins'].loc['HOU'] = 9
nfl_scores['Wins'].loc['WAS'] = 9
nfl_scores['Wins'].loc['ATL'] = 8
nfl_scores['Wins'].loc['BUF'] = 8
nfl_scores['Wins'].loc['IND'] = 8
nfl_scores['Wins'].loc['DET'] = 7
nfl_scores['Wins'].loc['NO'] = 7
nfl_scores['Wins'].loc['LV'] = 7
nfl_scores['Wins'].loc['PHI'] = 7
nfl_scores['Wins'].loc['LA'] = 7
nfl_scores['Wins'].loc['CHI'] = 6
nfl_scores['Wins'].loc['MIA'] = 6
nfl_scores['Wins'].loc['NYG'] = 6
nfl_scores['Wins'].loc['TB'] = 6
nfl_scores['Wins'].loc['BAL'] = 5
nfl_scores['Wins'].loc['JAX'] = 5
nfl_scores['Wins'].loc['SF'] = 5
nfl_scores['Wins'].loc['DAL'] = 4
nfl_scores['Wins'].loc['SD'] = 4
nfl_scores['Wins'].loc['CLE'] = 3
nfl_scores['Wins'].loc['TEN'] = 3

# print(nfl_scores.head())

print("Plotting wins vs Total predicted accuracy score:")
sns.lmplot(x='Wins',y='Total Accuracy score',data=nfl_scores,fit_reg=True)
plt.title('Wins vs Total Predicted Accuracy Score')
plt.show()
print("\n")

print("Plotting wins vs KNN predicted accuracy score:")
sns.lmplot(x='Wins',y='KNN score',data=nfl_scores,fit_reg=True)
plt.title('Wins vs KNN Predicted Accuracy Score')
plt.show()
print("\n")

print("Plotting wins vs SVM predicted accuracy score:")
sns.lmplot(x='Wins',y='SVM score',data=nfl_scores,fit_reg=True)
plt.title('Wins vs SVM Predicted Accuracy Score')
plt.show()
print("\n")

print("Plotting wins vs RF predicted accuracy score:")
sns.lmplot(x='Wins',y='RF score',data=nfl_scores,fit_reg=True)
plt.title('Wins vs Random Forest Predicted Accuracy Score')
plt.show()
print("\n")


### predicting short vs deep pass:

# pass_scores = pd.DataFrame(nfl_teams, columns = ['Team'])
# pass_scores = pass_scores.set_index('Team')
# pass_scores['Passing RF score'] = 0    

# pass_df = final_df
# pass_df = pass_df[pass_df['IsPass'] == 1]
# play_directions = ['SHORT RIGHT', 'SHORT LEFT', 'SHORT MIDDLE', 'DEEP RIGHT', 'DEEP LEFT', 'DEEP MIDDLE']
# pass_df = pass_df[pass_df['PassType'].isin(play_directions)]
# pass_df['IsDeep'] = pass_df.PassType.str.split().str.get(0)
# pass_df['IsDeep'] = pass_df["IsDeep"].str.contains("DEEP") * 1

# for team in nfl_teams:
#     team_df = pass_df[pass_df['OffenseTeam'] == team]
#     y = pass_df['IsDeep']
#     x = pass_df.drop(['OffenseTeam', 'IsPass', 'PlayType', 'PassType', 'RushDirection', 'IsDeep'], axis = 1)
    
#     xtrain, xtest, ytrain, ytest = train_test_split(x,y, test_size=0.2)   
#     # knn = KNeighborsClassifier(n_neighbors=10)
#     # knn.fit(xtrain, ytrain)
#     # passing_knn_score = knn.score(xtest, ytest)       
    
#     # #SVM Test
#     # svm = SVC(kernel="linear", C=0.025)
#     # svm.fit(xtrain, ytrain)
#     # passing_svm_score = svm.score(xtest, ytest)

#     #RF Test
#     forest = RandomForestClassifier(max_depth=5)
#     forest.fit(xtrain, ytrain)
#     passing_forest_score = forest.score(xtest, ytest)

#     pass_scores.loc[team] = [passing_forest_score]

# print(pass_scores.head())


# Repeat above predictions but with top 3 & 5 PCA components





























