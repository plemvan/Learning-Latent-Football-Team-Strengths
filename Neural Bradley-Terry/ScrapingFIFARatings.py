import pandas as pd
import numpy as np

# 1. Chargement
df = pd.read_csv("Data/ligue1_2010_2025.csv")

# 2. Préparation des colonnes intermédiaires
# On a besoin des Buts Marqués (Scored) et Encaissés (Conceded) pour chaque match
df['HomePts'] = df.apply(lambda x: 3 if x['FTR'] == 'H' else (1 if x['FTR'] == 'D' else 0), axis=1)
df['AwayPts'] = df.apply(lambda x: 3 if x['FTR'] == 'A' else (1 if x['FTR'] == 'D' else 0), axis=1)

# Pour l'équipe à Domicile : Scored = FTHG, Conceded = FTAG
df_home = df[['Season', 'HomeTeam', 'HomePts', 'FTHG', 'FTAG']].rename(
    columns={'HomeTeam': 'Team', 'HomePts': 'Points', 'FTHG': 'Goals_Scored', 'FTAG': 'Goals_Conceded'}
)
df_home['IsHome'] = 1 # Pour calculer le biais domicile plus tard

# Pour l'équipe à l'Extérieur : Scored = FTAG, Conceded = FTHG (inverse !)
df_away = df[['Season', 'AwayTeam', 'AwayPts', 'FTAG', 'FTHG']].rename(
    columns={'AwayTeam': 'Team', 'AwayPts': 'Points', 'FTAG': 'Goals_Scored', 'FTHG': 'Goals_Conceded'}
)
df_away['IsHome'] = 0

# Fusion
df_all = pd.concat([df_home, df_away], ignore_index=True)

# 3. Agrégation par Saison/Équipe
# On somme les points et les buts, et on compte le nombre de matchs
df_stats = df_all.groupby(['Season', 'Team'], as_index=False).agg({
    'Points': 'sum',
    'Goals_Scored': 'sum',
    'Goals_Conceded': 'sum',
    'IsHome': 'count' # Nombre de matchs joués
})
df_stats = df_stats.rename(columns={'IsHome': 'Games_Played'})

# Calcul séparé pour les points à Domicile uniquement (pour le biais)
home_points = df_all[df_all['IsHome'] == 1].groupby(['Season', 'Team'])['Points'].sum().reset_index()
home_points = home_points.rename(columns={'Points': 'Home_Points_Total'})

# Fusion des points domicile dans la table principale
df_stats = pd.merge(df_stats, home_points, on=['Season', 'Team'], how='left').fillna(0)

# =========================================================
# 4. CRÉATION DES VARIABLES AVANCÉES (FEATURE ENGINEERING)
# =========================================================

# A. Moyennes classiques
df_stats['Avg_Goals_Scored'] = df_stats['Goals_Scored'] / df_stats['Games_Played']
df_stats['Avg_Goals_Conceded'] = df_stats['Goals_Conceded'] / df_stats['Games_Played']
df_stats['Goal_Difference'] = df_stats['Goals_Scored'] - df_stats['Goals_Conceded']

# B. Pythagorean Expectation (Formule : G^2 / (G^2 + GA^2))
# C'est un excellent proxy de la "vraie" force, souvent meilleur que les points
df_stats['Pythagorean_Exp'] = (df_stats['Goals_Scored']**2) / (
    (df_stats['Goals_Scored']**2) + (df_stats['Goals_Conceded']**2)
)

# C. Home/Away Bias (Dépendance au domicile)
# Quelle part des points a été prise à la maison ?
# Si > 0.7, l'équipe voyage très mal.
df_stats['Home_Dependency'] = df_stats['Home_Points_Total'] / df_stats['Points']
# Sécurité division par zéro
df_stats['Home_Dependency'] = df_stats['Home_Dependency'].fillna(0.5) 

# D. Classement et Force (comme avant)
df_stats = df_stats.sort_values(by=['Season', 'Points'], ascending=[True, False])
df_stats['Rank'] = df_stats.groupby('Season')['Points'].rank(method='first', ascending=False).astype(int)
df_stats['Strength_Rank_Based'] = 1 - (df_stats['Rank'] / 21)

# =========================================================
# 5. RÉSULTAT FINAL
# =========================================================
# On garde les colonnes utiles pour le modèle
final_cols = [
    'Season', 'Team', 
    'Rank', 'Points', 
    'Strength_Rank_Based',   # Variable 1 : Basée sur le rang
    'Pythagorean_Exp',       # Variable 2 : Basée sur les buts (Qualité théorique)
    'Avg_Goals_Scored',      # Variable 3 : Puissance offensive
    'Avg_Goals_Conceded',    # Variable 4 : Solidité défensive
    'Home_Dependency'        # Variable 5 : Style de l'équipe
]

df_final = df_stats[final_cols]

print(df_final.head())
df_final.to_csv("table_enrichie_saisons.csv", index=False)