import pandas as pd
import numpy as np

# ==============================================================================
# 1. CHARGEMENT DES 3 FICHIERS
# ==============================================================================
# Adaptez les noms de fichiers selon ce que vous avez généré
df_matches = pd.read_csv("Data/Dataset_clean.csv")  # Fichier de matchs
df_stats = pd.read_csv("Data/table_enrichie_saisons.csv")
df_market = pd.read_csv("Data/ligue1_market_values_2010_2025.csv") # Ou le fichier Kaggle

# ==============================================================================
# 2. NETTOYAGE ET FORMATAGE DES SAISONS
# ==============================================================================

# A. Standardiser la Valeur Marchande
# Si votre fichier Market Value a des saisons format "2010-2011", il faut les transformer en "10-11"
# pour matcher avec votre fichier de matchs.
def format_market_season(season_str):
    # Transforme "2010-2011" en "10-11"
    # Transforme 2010 en "10-11" (si c'est un entier)
    try:
        if isinstance(season_str, int):
            start_year = season_str
        else:
            start_year = int(str(season_str).split('-')[0])
        
        # On ne garde que les 2 derniers chiffres
        s_start = start_year % 100
        s_end = (start_year + 1) % 100
        return f"{s_start:02d}-{s_end:02d}"
    except:
        return season_str

df_market['Season_Short'] = df_market['Season'].apply(format_market_season)
# On renomme pour faciliter la fusion
df_market = df_market[['Season_Short', 'Team', 'Total_Market_Value_Millions']]
df_market.columns = ['Season', 'Team', 'Market_Value']

# B. Créer la colonne "Saison Précédente" dans les Matchs
def get_prev_season(season):
    try:
        start = int(season.split('-')[0])
        prev_start = start - 1
        prev_end = start
        return f"{prev_start:02d}-{prev_end:02d}"
    except:
        return None

df_matches['Prev_Season'] = df_matches['Season'].apply(get_prev_season)

# ==============================================================================
# 3. FUSION 1 : AJOUT DE LA VALEUR MARCHANDE (Contexte Actuel)
# ==============================================================================
# On joint Match[Saison] avec Market[Saison]

# Pour Home Team
df_merged = pd.merge(
    df_matches, 
    df_market, 
    left_on=['Season', 'HomeTeam'], 
    right_on=['Season', 'Team'], 
    how='left'
).rename(columns={'Market_Value': 'Market_Value_Home'}).drop(columns=['Team'])

# Pour Away Team
df_merged = pd.merge(
    df_merged, 
    df_market, 
    left_on=['Season', 'AwayTeam'], 
    right_on=['Season', 'Team'], 
    how='left'
).rename(columns={'Market_Value': 'Market_Value_Away'}).drop(columns=['Team'])

# Remplissage des valeurs manquantes (pour les équipes non trouvées/petites)
# On met une valeur par défaut faible (ex: 10 Millions)
df_merged['Market_Value_Home'] = df_merged['Market_Value_Home'].fillna(10)
df_merged['Market_Value_Away'] = df_merged['Market_Value_Away'].fillna(10)

# ==============================================================================
# 4. FUSION 2 : AJOUT DE LA TABLE ENRICHIE (Historique N-1)
# ==============================================================================
# On joint Match[Prev_Season] avec Stats[Season]

# Colonnes à récupérer (Features historiques)
cols_to_keep = ['Season', 'Team', 'Strength_Rank_Based', 'Pythagorean_Exp', 'Avg_Goals_Scored', 'Home_Dependency']

# Pour Home Team
df_merged = pd.merge(
    df_merged,
    df_stats[cols_to_keep],
    left_on=['Prev_Season', 'HomeTeam'],
    right_on=['Season', 'Team'],
    how='left',
    suffixes=('', '_Stats')
).drop(columns=['Season_Stats', 'Team'])

# On renomme proprement
feature_names = ['Strength_Rank_Based', 'Pythagorean_Exp', 'Avg_Goals_Scored', 'Home_Dependency']
rename_dict_home = {col: col + '_Home' for col in feature_names}
df_merged = df_merged.rename(columns=rename_dict_home)

# Pour Away Team
df_merged = pd.merge(
    df_merged,
    df_stats[cols_to_keep],
    left_on=['Prev_Season', 'AwayTeam'],
    right_on=['Season', 'Team'],
    how='left',
    suffixes=('', '_Stats')
).drop(columns=['Season_Stats', 'Team'])

rename_dict_away = {col: col + '_Away' for col in feature_names}
df_merged = df_merged.rename(columns=rename_dict_away)

# ==============================================================================
# 5. GESTION DES PROMUBLES (Valeurs manquantes historiques)
# ==============================================================================
# Les équipes qui viennent de monter n'ont pas de stats pour N-1.
# On leur donne des valeurs de "Promu" (Faibles)

defaults = {
    'Strength_Rank_Based': 0.05,  # Considéré comme 20ème/21ème
    'Pythagorean_Exp': 0.30,      # Ratio de buts faible
    'Avg_Goals_Scored': 0.8,      # Marque peu
    'Home_Dependency': 0.5        # Neutre par défaut
}

for col in feature_names:
    df_merged[col + '_Home'] = df_merged[col + '_Home'].fillna(defaults[col])
    df_merged[col + '_Away'] = df_merged[col + '_Away'].fillna(defaults[col])

# ==============================================================================
# 6. FINALISATION ET SAUVEGARDE
# ==============================================================================
# Création de la Cible (Target)
# 1.0 = Home Win, 0.0 = Away Win, 0.5 = Draw
df_merged['Target'] = df_merged['FTR'].map({'H': 1.0, 'A': 0.0, 'D': 0.5})

# Normalisation Logarithmique pour la valeur marchande (Très important pour les réseaux de neurones)
# Car la différence entre 10M et 20M est plus importante qu'entre 500M et 510M
df_merged['Log_Market_Value_Home'] = np.log1p(df_merged['Market_Value_Home'])
df_merged['Log_Market_Value_Away'] = np.log1p(df_merged['Market_Value_Away'])

print("Dataset Final Prêt.")
print("Colonnes disponibles :", df_merged.columns)
print(df_merged[['Date', 'HomeTeam', 'Log_Market_Value_Home', 'Strength_Rank_Based_Home']].head())

df_merged.to_csv("dataset_final_training.csv", index=False)