#========== Neural Bradley-Terry Module ==========#

# Imports
import torch
import itertools
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.optim as optim

#=================================================#

# Definition of the neural Bradley-Terry model

class NeuralBradleyTerry(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 16, lr: float = 0.01, seed: int = 42):
        super(NeuralBradleyTerry, self).__init__()
        # Fix random seed for reproducibility
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1) # Output: latent strength score
        )
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.loss_fn = nn.BCEWithLogitsLoss()

    def forward(self, x):
        return self.feature_extractor(x)


    def fit(self, X_home, X_away, y, epochs=500):

        X_h = torch.FloatTensor(X_home)
        X_a = torch.FloatTensor(X_away)
        target = torch.FloatTensor(y).view(-1, 1)

        dataset = torch.utils.data.TensorDataset(X_h, X_a, target)
        loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)
        
        criterion = nn.BCEWithLogitsLoss() 
        optimizer = self.optimizer

        self.train()

        for epoch in range(epochs):
            for bh, ba, by in loader: 
                optimizer.zero_grad()
                score_h = self.forward(bh)
                score_a = self.forward(ba)
                loss = criterion(score_h - score_a, by)
                loss.backward()
                optimizer.step()


    def fit_seasons(self, X_home, X_away, y, seasons, epochs=300):
        
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
    

    def get_season_ranking(self, test_df : pd.DataFrame, feature_cols : list):

        """
        Generate team rankings for a specific season using the trained Neural Bradley-Terry model.
        """

        unique_teams = test_df.drop_duplicates(subset=['HomeTeam'])

        if unique_teams.empty:
            print("No data in test set")
            return None
        
        cols_home = [f"{col}_Home" for col in feature_cols]
        X_teams = unique_teams[cols_home].values.astype(np.float32)
        
        self.eval()
        with torch.no_grad():
            strengths = self.forward(torch.tensor(X_teams)).numpy().flatten()
        
        ranking = pd.DataFrame({
            'Team': unique_teams['HomeTeam'],
            'Neural_Strength': strengths
        })

        ranking['Neural_Strength'] = ranking['Neural_Strength'] - ranking['Neural_Strength'].mean()

        return ranking.sort_values(by='Neural_Strength', ascending=False)
    


    def predict_season_probas(self, test_df : pd.DataFrame, feature_cols : list, draw_threshold : float = 0.5, home_advantage : float = 0.0) -> pd.DataFrame:

        """
        Generate match outcome probabilities for all team matchups in a specific season.
        """

        unique_teams = test_df.drop_duplicates(subset=['HomeTeam'])

        if unique_teams.empty: return None

        cols_home = [f"{col}_Home" for col in feature_cols]
        X_teams = unique_teams[cols_home].values.astype(np.float32)
        
        self.eval()
        with torch.no_grad():
            strengths = self.forward(torch.tensor(X_teams)).numpy().flatten()
        
        team_strengths = dict(zip(unique_teams['HomeTeam'], strengths))
        
        # Simulation
        matches = []
        teams_list = list(team_strengths.keys())
        
        for home_team, away_team in itertools.permutations(teams_list, 2):
            s_h = team_strengths[home_team]
            s_a = team_strengths[away_team]
            
            # To handle draws, we use a threshold on the difference of strengths
            # P(Home) = P (Diff > threshold)
            prob_home = 1 / (1 + np.exp(-(s_h - s_a + home_advantage - draw_threshold)))
            
            # P(Away) = P (Diff < -threshold) 
            prob_away = 1 / (1 + np.exp(-(s_a - s_h - home_advantage - draw_threshold)))
            
            # P(Draw) = what is left
            prob_draw = max(0, 1 - prob_home - prob_away)
            
            total = prob_home + prob_draw + prob_away

            # Determine the most probable result
            if prob_home > prob_draw and prob_home > prob_away:
                result = 'H'
            elif prob_draw > prob_away:
                result = 'D'
            else:
                result = 'A'
            
            matches.append({
                'HomeTeam': home_team,
                'AwayTeam': away_team,
                'P_Home': prob_home / total,
                'P_Draw': prob_draw / total,
                'P_Away': prob_away / total,
                'Result': result,
                'Strength_Diff': s_h - s_a
            })
            
        return pd.DataFrame(matches)
