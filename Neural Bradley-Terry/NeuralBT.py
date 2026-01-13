import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# Definition of the neural Bradley-Terry model

class NeuralBradleyTerry(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 16, lr: float = 0.01):
        super(NeuralBradleyTerry, self).__init__()
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1) # Output: latent strength score
        )
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.loss_fn = nn.BCEWithLogitsLoss()

    def forward(self, x):
        return self.feature_extractor(x)

    def fit(self, X_home, X_away, y, seasons, epochs=300):
        
        seasons_arr = np.array([int(s.split('-')[0]) for s in seasons]) 
        min_year = seasons_arr.min()
        weights = (seasons_arr - min_year + 1) ** 2 
        weights = weights / weights.max() # Normalization between 0 and 1
        X_h = torch.FloatTensor(X_home)
        X_a = torch.FloatTensor(X_away)
        target = torch.FloatTensor(y).view(-1, 1)
        weights_tensor = torch.FloatTensor(weights).view(-1, 1)
        
        # Dataset with weights of the matches 
        dataset = torch.utils.data.TensorDataset(X_h, X_a, target, weights_tensor)
        loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)
        
        criterion = nn.BCEWithLogitsLoss(reduction='none') 
        optimizer = optim.Adam(self.parameters(), lr=0.01)
        
        self.train()
        print("Training with time-decayed weights")
        
        for epoch in range(epochs):
            for bh, ba, by, bw in loader: # bw = batch weights
                optimizer.zero_grad()
                score_h = self.forward(bh)
                score_a = self.forward(ba)
                raw_loss = criterion(score_h - score_a, by)
                weighted_loss = (raw_loss * bw).mean() # Weight applied here
                weighted_loss.backward()
                optimizer.step()

def get_season_ranking(model, df_full, target_season, feature_cols):
    """
    Generate team rankings for a specific season using the trained Neural Bradley-Terry model.
    """
    unique_teams = df_full[df_full['Season'] == target_season].drop_duplicates(subset=['HomeTeam'])
    if unique_teams.empty:
        print(f"Aucune donnée trouvée pour la saison {target_season}")
        return None
    
    cols_home = [f"{col}_Home" for col in feature_cols]
    X_teams = unique_teams[cols_home].values.astype(np.float32)
    
    model.eval()
    with torch.no_grad():
        strengths = model.forward(torch.tensor(X_teams)).numpy().flatten()
    
    ranking = pd.DataFrame({
        'Team': unique_teams['HomeTeam'],
        'Neural_Strength': strengths
    })

    return ranking.sort_values(by='Neural_Strength', ascending=False)

def get_start_year(s):
    return int(s.split('-')[0])

# Load processed dataset

df = pd.read_csv("Data/dataset_training.csv")
if 'Season_x' in df.columns:
    df = df.rename(columns={'Season_x': 'Season'})
df = df.dropna(subset=['Target'])

# Used features (every variable is explicitly named in the dataset)
feature_names = [
    'Log_Market_Value',     
    'Strength_Rank_Based',   
    'Pythagorean_Exp',       
    'Avg_Goals_Scored',      
    'Home_Dependency',
    'Avg_Shots_Target',  
    'Avg_Corners',       
    'Avg_Cards'        
]

target_season = '24-25' 

# Training data (all seasons before the target season)
df_train = df[df['Season'] != target_season].copy()
target_year = get_start_year(target_season)
df_train = df[df['Season'].apply(get_start_year) < target_year].copy()

# Info to be sure of the data used

print(f"Number of matches (Before {target_season}) : {len(df_train)}")
print(f"Used seasons : {df_train['Season'].unique()}")
seasons = df_train['Season'].values

X_home = df_train[[f"{col}_Home" for col in feature_names]].values.astype(np.float32)
X_away = df_train[[f"{col}_Away" for col in feature_names]].values.astype(np.float32)
y = df_train['Target'].values.astype(np.float32)

# Training loop and print results

model = NeuralBradleyTerry(input_dim=len(feature_names))
model.fit(X_home, X_away, y, seasons, epochs=500)

ranking = get_season_ranking(model, df, target_season, feature_names)

if ranking is not None:
    mean_strength = ranking['Neural_Strength'].mean()
    ranking['Centered_Strength'] = ranking['Neural_Strength'] - mean_strength # Centering around the mean
    plt.figure(figsize=(10, 8))
    df_plot = ranking.head(18).copy() #18 teams in the french league
    colors = ['#2ca02c' if x > 0 else '#d62728' for x in df_plot['Centered_Strength']]
    plt.barh(df_plot['Team'], df_plot['Centered_Strength'], color=colors)
    plt.axvline(0, color='black', linewidth=0.8, linestyle='--') 
    plt.gca().invert_yaxis() 
    plt.title(f"Relative strengths of the teams (Season {target_season})", fontsize=14)
    plt.xlabel("Latent strength", fontsize=12)
    for index, value in enumerate(df_plot['Centered_Strength']):
        plt.text(value, index, f"{value:.2f}", va='center', fontsize=9)
    plt.tight_layout()
    plt.show()