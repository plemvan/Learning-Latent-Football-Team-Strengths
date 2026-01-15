#========== Functions for NBTR Calibration ==========#

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

from Neural_Bradley_Terry.NeuralBT import NeuralBradleyTerry
from Data.data_processing import DataProcesser


class NBTR_calibrate:

    """
    Docstring for NBTR_calibrate
    """

    def __init__(self, train_data_filepath : str):

        """
        Docstring for __init__
        
        :param self: Description
        :param train_data_filepath: Description
        :type train_data_filepath: str
        """

        self.dp = DataProcesser(filepath=train_data_filepath)

        self.train_df , self.eval_df = self.dp.split_train_test(test_season='23-24')

        self.feature_names = [
            'Log_Market_Value',     
            'Strength_Rank_Based',   
            'Pythagorean_Exp',       
            'Avg_Goals_Scored',      
            'Home_Dependency',
            'Avg_Shots_Target',  
            'Avg_Corners',       
            'Avg_Cards'        
        ]

        self.train_data = self.dp.get_data_NBTR(feature_names=self.feature_names)

        # Prepare eval data similarly
        self.eval_data = {
            'X_home': self.eval_df[[f"{col}_Home" for col in self.feature_names]].values.astype(np.float32),
            'X_away': self.eval_df[[f"{col}_Away" for col in self.feature_names]].values.astype(np.float32),
            'y': self.eval_df['Target'].values.astype(np.float32),
            'input_dim': len(self.feature_names),
            'seasons': self.eval_df['Season'].values
        }
      
        return
    
    def calibrate_hyperparams(self, hidden_dims : list, lrs : list):
        """
        Calibrate hidden_dim and lr by minimizing BCE loss on evaluation data.
        
        Returns
        -------
        best_hidden_dim : int
            Optimal hidden dimension
        best_lr : float
            Optimal learning rate
        min_loss : float
            Minimum BCE loss achieved
        """        
               
        losses = np.zeros((len(hidden_dims), len(lrs)))
        
        for i, hd in enumerate(hidden_dims):
            for j, lr in enumerate(lrs):
                print(f"Training with hidden_dim={hd}, lr={lr}")
                
                # Create model
                model = NeuralBradleyTerry(input_dim=self.train_data['input_dim'], hidden_dim=hd, lr=lr)
                
                # Train on train data
                model.fit(self.train_data['X_home'], self.train_data['X_away'], self.train_data['y'], epochs=200)
                
                # Evaluate on eval data
                model.eval()
                with torch.no_grad():
                    score_h = model(torch.FloatTensor(self.eval_data['X_home']))
                    score_a = model(torch.FloatTensor(self.eval_data['X_away']))
                    y_eval = torch.FloatTensor(self.eval_data['y']).view(-1, 1)
                    loss = nn.BCEWithLogitsLoss()(score_h - score_a, y_eval)
                    losses[i, j] = loss.item()
        
        # Find best
        min_idx = np.unravel_index(np.argmin(losses), losses.shape)
        best_hidden_dim = hidden_dims[min_idx[0]]
        best_lr = lrs[min_idx[1]]
        min_loss = losses[min_idx]
        
        # Plot heatmap
        plt.figure(figsize=(8, 6))
        plt.imshow(losses, cmap='viridis', origin='lower')
        plt.colorbar(label='BCE Loss')
        plt.xticks(range(len(lrs)), lrs)
        plt.yticks(range(len(hidden_dims)), hidden_dims)
        plt.xlabel('Learning Rate')
        plt.ylabel('Hidden Dimension')
        plt.title('BCE Loss for Hyperparameter Grid')
        plt.scatter(min_idx[1], min_idx[0], color='red', marker='x', s=100, label=f'Best: hd={best_hidden_dim}, lr={best_lr}')
        plt.legend()
        plt.show()
        
        return best_hidden_dim, best_lr, min_loss
       