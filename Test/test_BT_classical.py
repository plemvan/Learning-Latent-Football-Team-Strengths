#========== Test/test_BT_classical.py ==========#

## Imports
import numpy as np
import pandas as pd
from typing import Literal
from scipy.special import expit
from scipy.stats import spearmanr, kendalltau
from BradleyTerry_classical.BT_classical import BradleyTerry


#===============================================#


class Comparison_Test:
    """Class used for the evaluation of the model"""


    def __init__(self, test_df : pd.DataFrame, result_BT : dict, result_NBTR : dict):

        # Test data
        self.test_df = test_df

        # Results
        self.result_BT = result_BT
        self.result_NBTR = result_NBTR

        if set(self.result_BT.keys()) == set(self.result_NBTR.keys()):
            
            self.teams = list(self.result_BT.keys())

        else:
            raise IndexError("Teams in each result dictionnary doesn't match")
            
        
        return
    
    def compute_proba(self, model : Literal['BT','NBTR'], team1 : str, team2 : str) -> dict:

        """
        Compute the probability of each outcome between two teams
        
        Parameters
        ----------
        model : str, 'BT' or 'NBTR'
            Which model to choose when computing probabilities
        team1 : str
            Name of first team
        team2 : str
            Name of second team
            
        Returns
        -------
        probs : dict
            Dictionary with keys 'team1_win', 'draw', 'team2_win'
        
        """

        # results
        if model == 'BT':
            results = self.result_BT
        else:
            results = self.result_NBTR

        # Strength of each team
        ti = results[team1]
        tj = results[team2]

        if model == 'BT':

            m = max


    

    def logLoss(self):

        """Compute the logLoss on test data"""

        logloss = 0.0

        for _, row in self.test_df.iterrows():

            team1 = row['HomeTeam']
            team2 = row['AwayTeam']
            result = row['FTR']  # 'H', 'D', 'A'

            probas = self.model.predict_proba(team1, team2)

            if result == 'H':
                logloss += -np.log(probas[f'{team1}_win'] + 1e-15)
            elif result == 'D':
                logloss += -np.log(probas['draw'] + 1e-15)
            elif result == 'A':
                logloss += -np.log(probas[f'{team2}_win'] + 1e-15)
            
        logloss /= len(self.test_df)

        print(f"LogLoss on test data: {logloss:.4f}")
        return logloss
    

    def baseline_logloss(self):

        """Compute the logLoss for a baseline model (uniform probabilities)"""

        logloss = 0.0

        for _, row in self.test_df.iterrows():

            team1 = row['HomeTeam']
            team2 = row['AwayTeam']
            result = row['FTR']  # 'H', 'D', 'A'

            # Uniform probabilities
            probas = {
                f'{team1}_win': 1/3,
                'draw': 1/3,
                f'{team2}_win': 1/3
            }

            if result == 'H':
                logloss += -np.log(probas[f'{team1}_win'] + 1e-15)
            elif result == 'D':
                logloss += -np.log(probas['draw'] + 1e-15)
            elif result == 'A':
                logloss += -np.log(probas[f'{team2}_win'] + 1e-15)

        logloss /= len(self.test_df)

        print(f"LogLoss on test data (baseline): {logloss:.4f}")
        return logloss