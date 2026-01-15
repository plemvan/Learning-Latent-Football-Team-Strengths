#========== Stability Analysis for NBTR ==========#

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import seaborn as sns
from scipy.stats import spearmanr

from .NeuralBT import NeuralBradleyTerry

class NBTR_Stability:
    """
    Class to analyze the stability of NBTR model across different random seeds.
    """

    def __init__(self, train_df : pd.DataFrame, test_df : pd.DataFrame, feature_names : list, nb_seed : int):
        """
        Initialize the stability analysis.

        Parameters
        ----------
        train_df : pd.DataFrame
            Training dataframe
        test_df : pd.DataFrame
            Test dataframe
        feature_names : list
            List of feature names
        nb_seed : int
            Number of different seeds to test
        """
        self.train_df = train_df
        self.test_df = test_df
        self.feature_names = feature_names
        self.nb_seed = nb_seed
        
        # Prepare data
        self.X_home = self.train_df[[f"{col}_Home" for col in self.feature_names]].values.astype(np.float32)
        self.X_away = self.train_df[[f"{col}_Away" for col in self.feature_names]].values.astype(np.float32)
        self.y = self.train_df['Target'].values.astype(np.float32)
        
        self.X_home_test = self.test_df[[f"{col}_Home" for col in self.feature_names]].values.astype(np.float32)
        self.X_away_test = self.test_df[[f"{col}_Away" for col in self.feature_names]].values.astype(np.float32)
        self.y_test = self.test_df['Target'].values.astype(np.float32)
        
        self.input_dim = len(self.feature_names)
        
        # Generate seeds
        np.random.seed(42)
        self.seeds = np.random.randint(0, 10000, self.nb_seed)
        
        self.models = []
        self.losses = []
        self.strengths = []
        self.ranks = []

    def run_analysis(self):
        """
        Run the stability analysis. Log-loss is computed on test data, strengths and ranks on train data.
        """
        # Get unique teams and their features
        unique_teams = self.train_df.drop_duplicates(subset=['HomeTeam'])
        cols_home = [f"{col}_Home" for col in self.feature_names]
        X_teams = unique_teams[cols_home].values.astype(np.float32)
        team_names = unique_teams['HomeTeam'].values
        
        for seed in self.seeds:
            print(f"Training with seed {seed}")
            
            # Create and train model
            model = NeuralBradleyTerry(input_dim=self.input_dim, hidden_dim=16, lr=0.01, seed=seed)
            model.fit(self.X_home, self.X_away, self.y, epochs=200)
            
            # Calculate log-loss (BCE) on test data
            model.eval()
            with torch.no_grad():
                score_h = model(torch.FloatTensor(self.X_home_test))
                score_a = model(torch.FloatTensor(self.X_away_test))
                y_tensor = torch.FloatTensor(self.y_test).view(-1, 1)
                loss = nn.BCEWithLogitsLoss()(score_h - score_a, y_tensor).item()
            
            self.losses.append(loss)
            self.models.append(model)
            
            # Get strengths
            model.eval()
            with torch.no_grad():
                strengths = model(torch.FloatTensor(X_teams)).numpy().flatten()
            self.strengths.append(strengths)
            
            # Get ranks (higher strength = better rank, so rank 1 is best)
            ranks = (-strengths).argsort().argsort() + 1  # rank starting from 1
            self.ranks.append(ranks)
        
        self.team_names = team_names
        self._analyze_losses()
        self._analyze_correlations()
        self._analyze_ranks()

    def _analyze_losses(self):
        """
        Analyze the log-losses.
        """
        losses = np.array(self.losses)
        print(f"Log-loss stats: mean={losses.mean():.4f}, std={losses.std():.4f}, min={losses.min():.4f}, max={losses.max():.4f}")
        
        # Boxplot
        plt.figure(figsize=(6, 4))
        plt.boxplot(losses)
        plt.title('Log-loss Distribution Across Seeds')
        plt.ylabel('BCE Loss')
        plt.show()

    def _analyze_correlations(self):
        """
        Analyze Spearman correlations between strengths.
        """
        n = len(self.strengths)
        corr_matrix = np.zeros((n, n))
        
        for i in range(n):
            for j in range(n):
                corr, _ = spearmanr(self.strengths[i], self.strengths[j])
                corr_matrix[i, j] = corr
        
        # Heatmap
        plt.figure(figsize=(8, 6))
        sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', vmin=-1, vmax=1)
        plt.title('Spearman Correlation Between Strengths Across Seeds')
        plt.xlabel('Seed Index')
        plt.ylabel('Seed Index')
        plt.show()

    def _analyze_ranks(self):
        """
        Analyze rank stability.
        """
        ranks = np.array(self.ranks)  # shape (nb_seed, n_teams)
        
        mean_ranks = ranks.mean(axis=0)
        std_ranks = ranks.std(axis=0)
        
        print("Top 5 most stable teams (lowest std):")
        stable_idx = np.argsort(std_ranks)[:5]
        for idx in stable_idx:
            print(f"{self.team_names[idx]}: mean rank {mean_ranks[idx]:.2f}, std {std_ranks[idx]:.2f}")
        
        print("\nTop 5 least stable teams (highest std):")
        unstable_idx = np.argsort(std_ranks)[-5:]
        for idx in unstable_idx:
            print(f"{self.team_names[idx]}: mean rank {mean_ranks[idx]:.2f}, std {std_ranks[idx]:.2f}")
        
        # Boxplots for most and least stable
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Most stable
        stable_data = [ranks[:, idx] for idx in stable_idx]
        axes[0].boxplot(stable_data, labels=[self.team_names[idx] for idx in stable_idx])
        axes[0].set_title('Rank Distribution for Most Stable Teams')
        axes[0].set_ylabel('Rank')
        
        # Least stable
        unstable_data = [ranks[:, idx] for idx in unstable_idx]
        axes[1].boxplot(unstable_data, labels=[self.team_names[idx] for idx in unstable_idx])
        axes[1].set_title('Rank Distribution for Least Stable Teams')
        axes[1].set_ylabel('Rank')
        
        plt.tight_layout()
        plt.show()