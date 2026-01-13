import pandas as pd
import numpy as np

# Functions for season standardization and previous season calculation and stats aggregation

def get_standard(s):
    """Convert all season formats to 'YY-YY' format."""
    s = str(s).strip()
    try:
        if len(s) > 5: 
            sep = '-' if '-' in s else '/'
            start = int(s.split(sep)[0])
            return f"{start % 100:02d}-{(start + 1) % 100:02d}"
        elif len(s) == 4 and s.isdigit(): 
            start = int(s)
            return f"{start % 100:02d}-{(start + 1) % 100:02d}"
        return s
    except:
        return s

def get_prev_season(season):
    """Compute the previous season string in 'YY-YY' format."""
    try:
        start = int(season.split('-')[0])
        prev_start = (start - 1) % 100
        prev_end = start
        return f"{prev_start:02d}-{prev_end:02d}"
    except:
        return None

def get_season_stats(df_matchs):
    """Calculate aggregated stats (Shots, Corners, Cards) per team per season."""

    home = df_matchs.groupby(['Season', 'HomeTeam']).agg({
        'HST': 'sum', 'HC': 'sum', 'HF': 'sum', 'HY': 'sum', 'HR': 'sum', 'FTR': 'count'
    }).reset_index() 
    home = home.rename(columns={'HomeTeam': 'Team', 'FTR': 'Games'})

    away = df_matchs.groupby(['Season', 'AwayTeam']).agg({
        'AST': 'sum', 'AC': 'sum', 'AF': 'sum', 'AY': 'sum', 'AR': 'sum', 'FTR': 'count'
    }).reset_index() 
    
    away = away.rename(columns={'AwayTeam': 'Team', 'FTR': 'Games', 
                                'AST': 'HST', 'AC': 'HC', 'AF': 'HF', 'AY': 'HY', 'AR': 'HR'}) 
    
    total = pd.concat([home, away])
    stats = total.groupby(['Season', 'Team']).sum().reset_index()
    
    stats['Avg_Shots_Target'] = stats['HST'] / stats['Games']
    stats['Avg_Corners'] = stats['HC'] / stats['Games']
    stats['Avg_Cards'] = (stats['HY'] + 3*stats['HR']) / stats['Games']
    
    return stats[['Season', 'Team', 'Avg_Shots_Target', 'Avg_Corners', 'Avg_Cards']]

# Loading and preprocessing datasets

df_matches = pd.read_csv("Data/Dataset_clean.csv")
df_stats = pd.read_csv("Data/features.csv")
df_market = pd.read_csv("Data/ligue1_market_values_2010_2025.csv") 

df_matches['Season'] = df_matches['Season'].astype(str).apply(get_standard)
df_stats['Season'] = df_stats['Season'].astype(str).apply(get_standard)
df_market['Season'] = df_market['Season'].astype(str).apply(get_standard)


if 'Total_Market_Value_Millions' in df_market.columns:
    df_market = df_market.rename(columns={'Total_Market_Value_Millions': 'Market_Value'})
elif 'Value' in df_market.columns:
    df_market = df_market.rename(columns={'Value': 'Market_Value'})

# Team Name Mapping
team_mapping = {
    "Ajaccio GFCO": "GFC Ajaccio", "Toulouse FC": "FC Toulouse", "Toulouse": "FC Toulouse",   
    "Arles": "AC Arles-Avignon", "Arles-Avignon": "AC Arles-Avignon", 
    "Evian Thonon Gaillard": "FC Évian Thonon Gaillard", "Evian TG": "Evian Thonon Gaillard FC",
    "Girondins Bordeaux": "FC Girondins Bordeaux", "Bordeaux": "FC Girondins Bordeaux",
    "Paris SG": "Paris Saint-Germain", "PSG": "Paris Saint-Germain",
    "St Etienne": "AS Saint-Étienne", "Saint-Étienne": "AS Saint-Étienne",
    "Lille": "LOSC Lille", "Lyon": "Olympique Lyon", "Marseille": "Olympique Marseille", 
    "Monaco": "AS Monaco", "Nice": "OGC Nice", "Rennes": "Stade Rennais FC", "Nantes": "FC Nantes",
    "Reims": "Stade Reims", "Montpellier": "Montpellier HSC", "Lorient": "FC Lorient",
    "Lens": "RC Lens", "Brest": "Stade Brestois 29", "Strasbourg": "RC Strasbourg Alsace", 
    "Metz": "FC Metz", "Troyes": "ESTAC Troyes", "Dijon": "Dijon FCO", "Angers": "Angers SCO", 
    "Ajaccio": "AC Ajaccio", "Gazelec Ajaccio": "GFC Ajaccio", "Bastia": "SC Bastia",
    "Nancy": "AS Nancy-Lorraine", "Caen": "SM Caen", "Guingamp": "EA Guingamp", 
    "Auxerre": "AJ Auxerre", "Sochaux": "FC Sochaux-Montbéliard", "Le Mans": "Le Mans FC",
    "Valenciennes": "Valenciennes FC", "Clermont": "Clermont Foot 63", "Amiens": "Amiens SC", 
    "Nimes": "Nîmes Olympique", "Le Havre": "Le Havre AC", "Boulogne": "US Boulogne"
}

for df in [df_matches, df_stats, df_market]:
    if 'HomeTeam' in df.columns: df['HomeTeam'] = df['HomeTeam'].replace(team_mapping)
    if 'AwayTeam' in df.columns: df['AwayTeam'] = df['AwayTeam'].replace(team_mapping)
    if 'Team' in df.columns: df['Team'] = df['Team'].replace(team_mapping)

df_matches['Prev_Season'] = df_matches['Season'].apply(get_prev_season)

# Merging datasets to create enriched training dataset

df_merged = pd.merge(df_matches, df_market[['Season', 'Team', 'Market_Value']], 
                     left_on=['Season', 'HomeTeam'], right_on=['Season', 'Team'], how='left')
df_merged = df_merged.rename(columns={'Market_Value': 'Market_Value_Home'}).drop(columns=['Team'])

df_merged = pd.merge(df_merged, df_market[['Season', 'Team', 'Market_Value']], 
                     left_on=['Season', 'AwayTeam'], right_on=['Season', 'Team'], how='left')
df_merged = df_merged.rename(columns={'Market_Value': 'Market_Value_Away'}).drop(columns=['Team'])

# Fill Market Value NaNs
df_merged['Market_Value_Home'] = df_merged['Market_Value_Home'].fillna(10)
df_merged['Market_Value_Away'] = df_merged['Market_Value_Away'].fillna(10)

# Historical Stats Enrichment
cols_stats = ['Season', 'Team', 'Strength_Rank_Based', 'Pythagorean_Exp', 'Avg_Goals_Scored', 'Home_Dependency']
feature_names = ['Strength_Rank_Based', 'Pythagorean_Exp', 'Avg_Goals_Scored', 'Home_Dependency']

# Merge Home Stats
df_merged = pd.merge(df_merged, df_stats[cols_stats], 
                     left_on=['Prev_Season', 'HomeTeam'], right_on=['Season', 'Team'], how='left')

if 'Season_x' in df_merged.columns: df_merged = df_merged.rename(columns={'Season_x': 'Season'})
if 'Season_y' in df_merged.columns: df_merged = df_merged.drop(columns=['Season_y'])
if 'Team' in df_merged.columns: df_merged = df_merged.drop(columns=['Team'])

rename_dict = {col: col + '_Home' for col in feature_names}
df_merged = df_merged.rename(columns=rename_dict)

df_merged = pd.merge(df_merged, df_stats[cols_stats], 
                     left_on=['Prev_Season', 'AwayTeam'], right_on=['Season', 'Team'], how='left')

# Cleaning
if 'Season_x' in df_merged.columns: df_merged = df_merged.rename(columns={'Season_x': 'Season'})
if 'Season_y' in df_merged.columns: df_merged = df_merged.drop(columns=['Season_y'])
if 'Team' in df_merged.columns: df_merged = df_merged.drop(columns=['Team'])

rename_dict = {col: col + '_Away' for col in feature_names}
df_merged = df_merged.rename(columns=rename_dict)

# Fill Stats NaNs (Promoted teams)
defaults = {'Strength_Rank_Based': 0.05, 'Pythagorean_Exp': 0.35, 'Avg_Goals_Scored': 0.8, 'Home_Dependency': 0.5}
for col in feature_names:
    df_merged[col + '_Home'] = df_merged[col + '_Home'].fillna(defaults[col])
    df_merged[col + '_Away'] = df_merged[col + '_Away'].fillna(defaults[col])

# Target Variables
df_merged['Target'] = df_merged['FTR'].map({'H': 1.0, 'A': 0.0, 'D': 0.5})
df_merged = df_merged.dropna(subset=['Target'])
df_merged['Log_Market_Value_Home'] = np.log1p(df_merged['Market_Value_Home'])
df_merged['Log_Market_Value_Away'] = np.log1p(df_merged['Market_Value_Away'])

#
df_game_stats = get_season_stats(df_merged)
df_merged = pd.merge(df_merged, df_game_stats, left_on=['Prev_Season', 'HomeTeam'], right_on=['Season', 'Team'], how='left')

# New cleaning 
if 'Season_x' in df_merged.columns: df_merged = df_merged.rename(columns={'Season_x': 'Season'})
if 'Season_y' in df_merged.columns: df_merged = df_merged.drop(columns=['Season_y'])
if 'Team' in df_merged.columns: df_merged = df_merged.drop(columns=['Team'])

df_merged = df_merged.rename(columns={
    'Avg_Shots_Target': 'Avg_Shots_Target_Home',
    'Avg_Corners': 'Avg_Corners_Home',
    'Avg_Cards': 'Avg_Cards_Home'
})

# Merge Away Game Stats
df_merged = pd.merge(df_merged, df_game_stats, left_on=['Prev_Season', 'AwayTeam'], right_on=['Season', 'Team'], how='left')

# New cleaning

if 'Season_x' in df_merged.columns: df_merged = df_merged.rename(columns={'Season_x': 'Season'})
if 'Season_y' in df_merged.columns: df_merged = df_merged.drop(columns=['Season_y'])
if 'Team' in df_merged.columns: df_merged = df_merged.drop(columns=['Team'])

df_merged = df_merged.rename(columns={
    'Avg_Shots_Target': 'Avg_Shots_Target_Away',
    'Avg_Corners': 'Avg_Corners_Away',
    'Avg_Cards': 'Avg_Cards_Away'
})

# Fill NaNs for Game Stats
cols_new = ['Avg_Shots_Target', 'Avg_Corners', 'Avg_Cards']
for c in cols_new:
    if f"{c}_Home" in df_merged.columns:
        mean_val = df_merged[f"{c}_Home"].mean()
        df_merged[f"{c}_Home"] = df_merged[f"{c}_Home"].fillna(mean_val)
        df_merged[f"{c}_Away"] = df_merged[f"{c}_Away"].fillna(mean_val)


output_file = "Data/dataset_training.csv"
df_merged.to_csv(output_file, index=False)
print("Processing complete. File saved to:", output_file)