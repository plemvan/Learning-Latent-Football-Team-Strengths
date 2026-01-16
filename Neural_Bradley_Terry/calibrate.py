#========== Module for calibrating NBTR Hyperparameters ==========#

## Imports
import torch
import numpy as np
import torch.nn as nn
from typing import Literal
import matplotlib.pyplot as plt
from sklearn.utils.class_weight import compute_class_weight

from Data.data_processing import DataProcesser
from Neural_Bradley_Terry.NeuralBT import NeuralBradleyTerry


#=================================================================#

class NBTR_calibrate:

    """Class for calibrating NBTR Hyperparameters"""

    def __init__(self, train_data_filepath : str, eval_season : str = '23-24'):

        """
        Calibrating NBTR Hyperparameters

        Parameters:
        -----------
        train_data_filepath : str
            Filepath to the training data CSV
        eval_season : str, default='23-24'
            Season to use for evaluation
        """
        
        # Data Processing
        self.dp = DataProcesser(filepath=train_data_filepath)

        self.train_df , self.eval_df = self.dp.split_train_test(test_season=eval_season)

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

        # Prepare evaluation data similarly
        self.eval_data = {
            'X_home': self.eval_df[[f"{col}_Home" for col in self.feature_names]].values.astype(np.float32),
            'X_away': self.eval_df[[f"{col}_Away" for col in self.feature_names]].values.astype(np.float32),
            'y': self.eval_df['Target'].values.astype(np.float32),
            'input_dim': len(self.feature_names),
            'seasons': self.eval_df['Season'].values
        }

        # Best param
        self.hidden_dim : int = None
        self.lr : float = None
      
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
        """        
               
        losses = np.zeros((len(hidden_dims), len(lrs)))
        
        for i, hd in enumerate(hidden_dims):
            for j, lr in enumerate(lrs):
                print(f"Training with hidden_dim={hd}, lr={lr}")
                
                # Create model
                model = NeuralBradleyTerry(input_dim=self.train_data['input_dim'], hidden_dim=hd, lr=lr)
                
                # Train on training set
                model.fit(self.train_data['X_home'], self.train_data['X_away'], self.train_data['y'], epochs=200)
                
                # Evaluate on evaluation set
                model.eval()
                with torch.no_grad():
                    score_h = model(torch.FloatTensor(self.eval_data['X_home']))
                    score_a = model(torch.FloatTensor(self.eval_data['X_away']))
                    y_eval = torch.FloatTensor(self.eval_data['y']).view(-1, 1)
                    loss = nn.BCEWithLogitsLoss()(score_h - score_a, y_eval)
                    losses[i, j] = loss.item()
        
        # Find optimal values
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

        self.hidden_dim = best_hidden_dim
        self.lr = best_lr

        print(f"Optimal hidden_dim: {best_hidden_dim}")
        print(f"Optimal lr: {best_lr}")
        print(f"Minimum loss: {min_loss:.4f}")
        
        return best_hidden_dim, best_lr
       

    def calibrate_draw_threshold(self, draw_thresholds : list, epochs : int = 200):

        """
        Calibrate draw_threshold parameter by minimizing log loss (for NBTR.fit (without season)).
        
        Parameters
        ----------
        draw_thresholds : list
            List of draw threshold values to test
        epochs : int, default=200
            Number of training epochs
            
        Returns
        -------
        best_threshold : float
            Optimal draw threshold
        """

        # Hyperparameters for training
        hidden_dim = self.hidden_dim if self.hidden_dim is not None else 16
        lr = self.lr if self.lr is not None else 0.01
        
        # Train model once
        print("Training model...")
        model = NeuralBradleyTerry(input_dim=self.train_data['input_dim'], hidden_dim=hidden_dim, lr=lr)
        model.fit(self.train_data['X_home'], self.train_data['X_away'], self.train_data['y'], epochs=epochs)
        print("End of training")
        
        # Weights (unused, all set to 1)
        class_weights = {'H': 1.0, 'D': 1.0, 'A': 1.0}
        
        # Test each threshold
        losses = []
        epsilon = 1e-15  # For numerical stability
        
        for threshold in draw_thresholds:
            
            probas_df = model.predict_season_probas(
                test_df=self.eval_df,
                feature_cols=self.feature_names,
                draw_threshold=threshold,
                home_advantage=0.0
            )
            
            log_loss_sum = 0.0
            
            proba_dict = {}
            for _, row in probas_df.iterrows():

                key = (row['HomeTeam'], row['AwayTeam'])
                proba_dict[key] = {
                    'P_Home': row['P_Home'],
                    'P_Draw': row['P_Draw'],
                    'P_Away': row['P_Away']
                }
            
            # Compute log loss for each match in eval_df
            for _, row in self.eval_df.iterrows():

                key = (row['HomeTeam'], row['AwayTeam'])
                true_result = row['FTR']
                weight = class_weights[true_result]
                
                probas = proba_dict[key]
                
                # Compute log loss for this sample
                if true_result == 'H':
                    log_loss_sum += -weight * np.log(max(probas['P_Home'], epsilon))
                elif true_result == 'D':
                    log_loss_sum += -weight * np.log(max(probas['P_Draw'], epsilon))
                else:  # 'A'
                    log_loss_sum += -weight * np.log(max(probas['P_Away'], epsilon))
            
            # Average (weighted) log loss
            weighted_log_loss = log_loss_sum / len(self.eval_df)
            losses.append(weighted_log_loss)
        
        # Find optimal threshold
        min_idx = np.argmin(losses)
        best_threshold = draw_thresholds[min_idx]
        min_loss = losses[min_idx]
        
        # Plot
        plt.figure(figsize=(10, 6))
        plt.plot(draw_thresholds, losses, linewidth=2)
        plt.axvline(best_threshold, color='red', linestyle='--', label=f'Optimal: {best_threshold:.3f}')
        plt.xlabel('Draw Threshold')
        plt.ylabel('Weighted Log Loss')
        plt.title('Weighted Log Loss vs Draw Threshold')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.show()

        print(f"Best draw_threshold: {best_threshold:.3f}")
        print(f"Minimum loss: {min_loss:.4f}")
        
        return best_threshold
    
    
    def calibrate_threshold_and_advantage(self, draw_thresholds : list, home_advantages : list, epochs : int = 300):

        """
        Calibrate both draw_threshold and home_advantage by minimizing log loss (for NBTR.fit_season). 
        
        Parameters
        ----------
        draw_thresholds : list
            List of draw threshold values to test
        home_advantages : list
            List of home advantage values to test
        epochs : int, default=300
            Number of training epochs
            
        Returns
        -------
        best_threshold : float
            Optimal draw threshold
        best_advantage : float
            Optimal home advantage
        """

        # Hyperparameters for training
        hidden_dim = self.hidden_dim if self.hidden_dim is not None else 16
        lr = self.lr if self.lr is not None else 0.01
        
        # Train model once with time-weighted training
        print("Training model (with time-decayed weights)...")
        model = NeuralBradleyTerry(input_dim=self.train_data['input_dim'], hidden_dim=hidden_dim, lr=lr)
        model.fit_seasons(self.train_data['X_home'], self.train_data['X_away'], 
                         self.train_data['y'], self.train_data['seasons'], epochs=epochs)
        print("End of training")

        # Weights (unused, all set to 1)
        class_weights = {'H': 1.0, 'D': 1.0, 'A': 1.0}
        
        # Grid search
        losses = np.zeros((len(draw_thresholds), len(home_advantages)))
        epsilon = 1e-15  # For numerical stability
        
        for i, threshold in enumerate(draw_thresholds):
            for j, advantage in enumerate(home_advantages):
                
                probas_df = model.predict_season_probas(
                    test_df=self.eval_df,
                    feature_cols=self.feature_names,
                    draw_threshold=threshold,
                    home_advantage=advantage
                )
                
                log_loss_sum = 0.0
                
                proba_dict = {}
                for _, row in probas_df.iterrows():

                    key = (row['HomeTeam'], row['AwayTeam'])
                    proba_dict[key] = {
                        'P_Home': row['P_Home'],
                        'P_Draw': row['P_Draw'],
                        'P_Away': row['P_Away']
                    }
                
                # Compute log loss for each match in eval_df
                for _, row in self.eval_df.iterrows():

                    key = (row['HomeTeam'], row['AwayTeam'])
                    true_result = row['FTR']
                    weight = class_weights[true_result]
                    
                    probas = proba_dict[key]
                    
                    # Compute log loss for this sample
                    if true_result == 'H':
                        log_loss_sum += -weight * np.log(max(probas['P_Home'], epsilon))
                    elif true_result == 'D':
                        log_loss_sum += -weight * np.log(max(probas['P_Draw'], epsilon))
                    else:  # 'A'
                        log_loss_sum += -weight * np.log(max(probas['P_Away'], epsilon))
                
                # Average (weighted) log loss
                losses[i, j] = log_loss_sum / len(self.eval_df)
        
        # Find optimal parameters
        min_idx = np.unravel_index(np.argmin(losses), losses.shape)
        best_threshold = draw_thresholds[min_idx[0]]
        best_advantage = home_advantages[min_idx[1]]
        min_loss = losses[min_idx]
        
        # Plot heatmap
        plt.figure(figsize=(10, 8))
        plt.imshow(losses, cmap='viridis', origin='lower', aspect='auto')
        plt.colorbar(label='Weighted Log Loss')
        plt.xticks(range(0,len(home_advantages),np.floor(len(home_advantages)/10)), [f'{ha:.2f}' for ha in home_advantages])
        plt.yticks(range(0,len(draw_thresholds),np.floor(len(draw_thresholds)/10)), [f'{dt:.2f}' for dt in draw_thresholds])
        plt.xlabel('Home Advantage')
        plt.ylabel('Draw Threshold')
        plt.title(f'Weighted Log Loss Heatmap')
        plt.scatter(min_idx[1], min_idx[0], color='red', marker='x', s=200, linewidths=3,
                   label=f'Best: dt={best_threshold:.2f}, ha={best_advantage:.2f}')
        plt.legend()
        plt.tight_layout()
        plt.show()
        
        print(f"Best draw_threshold: {best_threshold:.3f}")
        print(f"Best home_advantage: {best_advantage:.3f}")
        print(f"Minimum loss: {min_loss:.4f}")
        
        return best_threshold, best_advantage