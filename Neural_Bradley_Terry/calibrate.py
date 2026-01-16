#========== Functions for NBTR Calibration ==========#

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from sklearn.utils.class_weight import compute_class_weight
from typing import Literal


from Neural_Bradley_Terry.NeuralBT import NeuralBradleyTerry
from Data.data_processing import DataProcesser


class NBTR_calibrate:

    """
    Docstring for NBTR_calibrate
    """

    def __init__(self, train_data_filepath : str, eval_season : str = '23-24'):

        """
        Docstring for __init__
        
        :param self: Description
        :param train_data_filepath: Description
        :type train_data_filepath: str
        """

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
       

    def calibrate_draw_threshold(self, draw_thresholds : list, weight_mode : Literal['manual','inverse_freq'] = 'manual', 
                                   draw_weight : float = 2.0, hidden_dim : int = 16, lr : float = 0.01, epochs : int = 200):
        """
        Calibrate draw_threshold parameter using classic fit method by minimizing weighted log loss.
        
        Parameters
        ----------
        draw_thresholds : list
            List of draw threshold values to test
        weight_mode : Literal['manual','inverse_freq'], default='manual'
            Weighting mode: 'manual' or 'inverse_freq'
        draw_weight : float, default=2.0
            Weight for draw outcomes when weight_mode='manual' (H and A get weight 1.0)
        hidden_dim : int, default=16
            Hidden dimension for the model
        lr : float, default=0.01
            Learning rate for the model
        epochs : int, default=200
            Number of training epochs
            
        Returns
        -------
        best_threshold : float
            Optimal draw threshold
        min_loss : float
            Minimum weighted log loss achieved
        """
        
        # Train model once
        print("Training model...")
        model = NeuralBradleyTerry(input_dim=self.train_data['input_dim'], hidden_dim=hidden_dim, lr=lr)
        model.fit(self.train_data['X_home'], self.train_data['X_away'], self.train_data['y'], epochs=epochs)
        
        # Compute class weights
        if weight_mode == 'inverse_freq':
            unique_classes = np.unique(self.eval_df['FTR'])
            class_weights_array = compute_class_weight('balanced', classes=unique_classes, y=self.eval_df['FTR'])
            class_weights = dict(zip(unique_classes, class_weights_array))
            print(f"Inverse frequency weights: {class_weights}")
        elif weight_mode == 'manual':
            class_weights = {'H': 1.0, 'D': draw_weight, 'A': 1.0}
            print(f"Manual weights: {class_weights}")
        else:
            raise ValueError("'weight_mode' should be 'inverse_freq' or 'manual'")

        class_weights = {'H': 1.0, 'D': 1.0, 'A': 1.0}
        
        # Test each threshold
        losses = []
        epsilon = 1e-15  # For numerical stability
        
        for threshold in draw_thresholds:
            # Use predict_season_probas to get probabilities
            probas_df = model.predict_season_probas(
                test_df=self.eval_df,
                feature_cols=self.feature_names,
                draw_threshold=threshold,
                home_advantage=0.0
            )
            
            log_loss_sum = 0.0
            
            # Create a mapping from (HomeTeam, AwayTeam) to probabilities
            proba_dict = {}
            for _, row in probas_df.iterrows():
                key = (row['HomeTeam'], row['AwayTeam'])
                proba_dict[key] = {
                    'P_Home': row['P_Home'],
                    'P_Draw': row['P_Draw'],
                    'P_Away': row['P_Away']
                }
            
            # Compute log loss for each match in eval_df
            for idx, row in self.eval_df.iterrows():
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
            
            # Average weighted log loss
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
        plt.title(f'Weighted Log Loss vs Draw Threshold (mode={weight_mode})')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.show()
        
        print(f"Best draw_threshold: {best_threshold:.3f} with loss: {min_loss:.4f}")
        
        return best_threshold, min_loss
    
    
    def calibrate_threshold_and_advantage(self, draw_thresholds : list, home_advantages : list, 
                                           weight_mode : Literal['manual','inverse_freq'] = 'manual', draw_weight : float = 2.0,
                                           hidden_dim : int = 16, lr : float = 0.01, epochs : int = 300):
        """
        Calibrate both draw_threshold and home_advantage using fit_seasons method by minimizing weighted log loss.
        
        Parameters
        ----------
        draw_thresholds : list
            List of draw threshold values to test
        home_advantages : list
            List of home advantage values to test
        weight_mode : Literal['manual','inverse_freq'], default='manual'
            Weighting mode: 'manual' or 'inverse_freq'
        draw_weight : float, default=2.0
            Weight for draw outcomes when weight_mode='manual'
        hidden_dim : int, default=16
            Hidden dimension for the model
        lr : float, default=0.01
            Learning rate for the model
        epochs : int, default=300
            Number of training epochs
            
        Returns
        -------
        best_threshold : float
            Optimal draw threshold
        best_advantage : float
            Optimal home advantage
        min_loss : float
            Minimum weighted log loss achieved
        """
        
        # Train model once with time-weighted training
        print("Training model with time-decayed weights...")
        model = NeuralBradleyTerry(input_dim=self.train_data['input_dim'], hidden_dim=hidden_dim, lr=lr)
        model.fit_seasons(self.train_data['X_home'], self.train_data['X_away'], 
                         self.train_data['y'], self.train_data['seasons'], epochs=epochs)
        
        # Compute class weights
        if weight_mode == 'inverse_freq':
            unique_classes = np.unique(self.eval_df['FTR'])
            class_weights_array = compute_class_weight('balanced', classes=unique_classes, y=self.eval_df['FTR'])
            class_weights = dict(zip(unique_classes, class_weights_array))
            print(f"Inverse frequency weights: {class_weights}")
        elif weight_mode == 'manual':
            class_weights = {'H': 1.0, 'D': draw_weight, 'A': 1.0}
            print(f"Manual weights: {class_weights}")
        else:
            raise ValueError("'weight_mode' should be 'inverse_freq' or 'manual'")

        class_weights = {'H': 1.0, 'D': 1.0, 'A': 1.0}
        
        # Grid search
        losses = np.zeros((len(draw_thresholds), len(home_advantages)))
        epsilon = 1e-15  # For numerical stability
        
        for i, threshold in enumerate(draw_thresholds):
            for j, advantage in enumerate(home_advantages):
                
                # Use predict_season_probas to get probabilities
                probas_df = model.predict_season_probas(
                    test_df=self.eval_df,
                    feature_cols=self.feature_names,
                    draw_threshold=threshold,
                    home_advantage=advantage
                )
                
                log_loss_sum = 0.0
                
                # Create a mapping from (HomeTeam, AwayTeam) to probabilities
                proba_dict = {}
                for _, row in probas_df.iterrows():
                    key = (row['HomeTeam'], row['AwayTeam'])
                    proba_dict[key] = {
                        'P_Home': row['P_Home'],
                        'P_Draw': row['P_Draw'],
                        'P_Away': row['P_Away']
                    }
                
                # Compute log loss for each match in eval_df
                for idx, row in self.eval_df.iterrows():
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
                
                # Average weighted log loss
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
        plt.xticks(range(len(home_advantages)), [f'{ha:.2f}' for ha in home_advantages])
        plt.yticks(range(len(draw_thresholds)), [f'{dt:.2f}' for dt in draw_thresholds])
        plt.xlabel('Home Advantage')
        plt.ylabel('Draw Threshold')
        plt.title(f'Weighted Log Loss Heatmap (mode={weight_mode})')
        plt.scatter(min_idx[1], min_idx[0], color='red', marker='x', s=200, linewidths=3,
                   label=f'Best: dt={best_threshold:.2f}, ha={best_advantage:.2f}')
        plt.legend()
        plt.tight_layout()
        plt.show()
        
        print(f"Best draw_threshold: {best_threshold:.3f}")
        print(f"Best home_advantage: {best_advantage:.3f}")
        print(f"Minimum loss: {min_loss:.4f}")
        
        return best_threshold, best_advantage, min_loss