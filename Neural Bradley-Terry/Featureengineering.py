import pandas as pd

#Config

df = pd.read_csv("Data/ligue1_2010_2025.csv")

df['HomePts'] = df.apply(lambda x: 3 if x['FTR'] == 'H' else (1 if x['FTR'] == 'D' else 0), axis=1)
df['AwayPts'] = df.apply(lambda x: 3 if x['FTR'] == 'A' else (1 if x['FTR'] == 'D' else 0), axis=1)

df_home = df[['Season', 'HomeTeam', 'HomePts', 'FTHG', 'FTAG']].rename(
    columns={'HomeTeam': 'Team', 'HomePts': 'Points', 'FTHG': 'Goals_Scored', 'FTAG': 'Goals_Conceded'}
)

df_home['IsHome'] = 1 
df_away = df[['Season', 'AwayTeam', 'AwayPts', 'FTAG', 'FTHG']].rename(
    columns={'AwayTeam': 'Team', 'AwayPts': 'Points', 'FTAG': 'Goals_Scored', 'FTHG': 'Goals_Conceded'}
)

df_away['IsHome'] = 0
df_all = pd.concat([df_home, df_away], ignore_index=True)


df_stats = df_all.groupby(['Season', 'Team'], as_index=False).agg({
    'Points': 'sum',
    'Goals_Scored': 'sum',
    'Goals_Conceded': 'sum',
    'IsHome': 'count' 
})
df_stats = df_stats.rename(columns={'IsHome': 'Games_Played'})

home_points = df_all[df_all['IsHome'] == 1].groupby(['Season', 'Team'])['Points'].sum().reset_index()
home_points = home_points.rename(columns={'Points': 'Home_Points_Total'})

df_stats = pd.merge(df_stats, home_points, on=['Season', 'Team'], how='left').fillna(0)

# Feature engineering part

# Classical means 
df_stats['Avg_Goals_Scored'] = df_stats['Goals_Scored'] / df_stats['Games_Played']
df_stats['Avg_Goals_Conceded'] = df_stats['Goals_Conceded'] / df_stats['Games_Played']
df_stats['Goal_Difference'] = df_stats['Goals_Scored'] - df_stats['Goals_Conceded']

#  Pythagorean Expectation (Formule : G^2 / (G^2 + GA^2))
df_stats['Pythagorean_Exp'] = (df_stats['Goals_Scored']**2) / (
    (df_stats['Goals_Scored']**2) + (df_stats['Goals_Conceded']**2)
)

# Home dependency
df_stats['Home_Dependency'] = df_stats['Home_Points_Total'] / df_stats['Points']
df_stats['Home_Dependency'] = df_stats['Home_Dependency'].fillna(0.5) 

# Ranks in the previous seasons
df_stats = df_stats.sort_values(by=['Season', 'Points'], ascending=[True, False])
df_stats['Rank'] = df_stats.groupby('Season')['Points'].rank(method='first', ascending=False).astype(int)
df_stats['Strength_Rank_Based'] = 1 - (df_stats['Rank'] / 21)

final_cols = [
    'Season', 'Team', 
    'Rank', 'Points', 
    'Strength_Rank_Based', 
    'Pythagorean_Exp',       
    'Avg_Goals_Scored',      
    'Avg_Goals_Conceded',    
    'Home_Dependency'        
]

df_final = df_stats[final_cols]
print(df_final.head())
df_final.to_csv("features.csv", index=False)