import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# ==============================================================================
# 1. DÉFINITION DU MODÈLE (Neural Bradley-Terry)
# ==============================================================================
class NeuralBradleyTerry(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 16, lr: float = 0.01):
        super(NeuralBradleyTerry, self).__init__()
        # Architecture simple : Features -> Force
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1) # Sortie : 1 seul chiffre (la force)
        )
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.loss_fn = nn.BCEWithLogitsLoss()

    def forward(self, x):
        return self.feature_extractor(x)

    def fit(self, X_home, X_away, y, epochs=300):
        # Conversion en Tensor
        X_h = torch.FloatTensor(X_home)
        X_a = torch.FloatTensor(X_away)
        target = torch.FloatTensor(y).view(-1, 1)
        
        self.train()
        print("Début de l'entraînement...")
        
        for epoch in range(epochs):
            self.optimizer.zero_grad()
            
            # Calcul des forces
            score_h = self.forward(X_h)
            score_a = self.forward(X_a)
            
            # Bradley-Terry : la différence de force prédit la victoire
            logits = score_h - score_a 
            
            loss = self.loss_fn(logits, target)
            loss.backward()
            self.optimizer.step()
            
            if epoch % 50 == 0:
                print(f"Epoch {epoch} | Loss: {loss.item():.4f}")
        
        print("Entraînement terminé.")


if __name__ == '__main__':
    
    # ==============================================================================
    # 2. CHARGEMENT ET PRÉPARATION DES DONNÉES
    # ==============================================================================
    # On charge le fichier final créé précédemment
    df = pd.read_csv("Data/dataset_final_training.csv")

    # Liste des features (Assurez-vous que ces colonnes existent dans votre CSV)
    # Note : on utilise les noms de base, on rajoutera _Home et _Away après
    feature_names = [
    'Log_Market_Value',      # Structurel
    'Strength_Rank_Based',   # Historique
    'Pythagorean_Exp',       # Forme théorique
    'Avg_Goals_Scored',      # Attaque
    'Home_Dependency'        # Domicile
    ]

    # Création des matrices X_home, X_away et y
    # On prend toutes les données (Train + Test mélangés pour cet exemple simple)
    X_home = df[[f"{col}_Home" for col in feature_names]].values.astype(np.float32)
    X_away = df[[f"{col}_Away" for col in feature_names]].values.astype(np.float32)
    y = df['Target'].values.astype(np.float32)

    # ==============================================================================
    # 3. ENTRAÎNEMENT
    # ==============================================================================
    model = NeuralBradleyTerry(input_dim=len(feature_names))
    model.fit(X_home, X_away, y, epochs=500)

    # ==============================================================================
    # 4. OBTENIR LES FORCES (Ce que vous demandez)
    # ==============================================================================

    def get_season_ranking(model, df_full, target_season, feature_cols):
        """
        Extrait les forces des équipes pour une saison précise.
        """
        # 1. On filtre pour garder une seule ligne par équipe pour cette saison
        # On utilise les infos "Home" pour avoir les stats de l'équipe
        unique_teams = df_full[df_full['Season'] == target_season].drop_duplicates(subset=['HomeTeam'])
    
        if unique_teams.empty:
            print(f"Aucune donnée trouvée pour la saison {target_season}")
            return None

        # 2. Préparation des features de ces équipes
        cols_home = [f"{col}_Home" for col in feature_cols]
        X_teams = unique_teams[cols_home].values.astype(np.float32)
    
        # 3. Le modèle prédit la force !
        model.eval()
        with torch.no_grad():
            strengths = model.forward(torch.tensor(X_teams)).numpy().flatten()
        
            strengths = strengths - np.mean(strengths)  # Centrage pour interprétation relative

            print(f"Moyennes des forces {np.mean(strengths)}")
    
        # 4. Création du tableau de résultat
        ranking = pd.DataFrame({
            'Team': unique_teams['HomeTeam'],
            'Neural_Strength': strengths
        })
    
        # On trie du plus fort au plus faible
        return ranking.sort_values(by='Neural_Strength', ascending=False)

    # --- EXEMPLE : CLASSEMENT POUR LA DERNIÈRE SAISON DU FICHIER ---
    last_season = '2024-2025' # Ex: "22-23" ou "2022" selon votre format
    print(f"\n--- CLASSEMENT NEURAL POUR LA SAISON {last_season} ---")

    ranking = get_season_ranking(model, df, last_season, feature_names)

    if ranking is not None:
        print(ranking.head(10)) # Affiche le Top 10

        # Petit graphique pour votre rapport
        plt.figure(figsize=(10, 6))
        # On prend le top 15 pour lisibilité
        top_15 = ranking.head(18)
        plt.barh(top_15['Team'], top_15['Neural_Strength'], color='skyblue')
        plt.gca().invert_yaxis() # Le 1er en haut
        plt.xlabel("Latent Strength learned by NeuralBT")
        plt.title(f"Ranking  - Season {last_season}")
        plt.show()