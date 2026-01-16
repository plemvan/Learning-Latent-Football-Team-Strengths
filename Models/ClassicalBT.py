#========== Module for Classical Bradley-Terry Model ==========#

## Imports
import numpy as np
import pandas as pd
from ModelTools.Gradient_descent import GradientDescent


#==============================================================#

class BradleyTerry():

    """Class implementing the classical Bradley-Terry framework"""

    def __init__(self, lambda_draw : float, learning_rate : float = 0.01, n_iterations : int = 1000, tolerance : float = 1e-6):
        
        """
        Bradley-Terry model

        Parameters
        ----------

        lambda_draw : float
            Draw parameter
        learning_rate : float. Default = 0.01
            Step of the gradient descent
        n_iterations : int. Default = 1000
            Number of iterations of the algorithm
        tolerance : float. Default = 1e-6
            Stop criterion (if the change is less than this threshold)
        
        """

        # Hyperparameters for the gradient descent
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.tolerance = tolerance

        # Draw Hyperparameter
        self.lambda_draw = lambda_draw

        # Attributes
        self.W : np.array = None # Matrix of victories
        self.D : np.array = None # Matrix of draws

        self.theta : np.array = None # Vector of strengths

        self.teams = None
        self.teams_index : dict = None

        return
    

    def loglikelihood(self, theta : np.array)-> float:

        """
        Compute and return the log-likelihood of the model for a vector of strengths theta

        Parameters
        ----------
        theta : np.array
            Vector of strengths

        Returns
        -------
        loglik : float
            Value of the log-likelihood for the given X and theta
        """

        W = self.W
        D = self.D
        lambda_draw = self.lambda_draw

        theta = np.asarray(theta)
        n = len(theta)

        loglik = 0.0

        for i in range(n):
            for j in range(i+1, n):

                ti = theta[i]
                tj = theta[j]

                # linear (observed) terms
                loglik += W[i, j] * ti
                loglik += W[j, i] * tj
                loglik += 0.5 * D[i, j] * (ti + tj)
                loglik += D[i, j] * lambda_draw

                # total number of games
                Nij = W[i, j] + W[j, i] + D[i, j]

                # numerically stable log Z_ij
                m = max(ti, tj)
                logZ = (
                    m
                    + np.log(
                        np.exp(ti - m)
                        + np.exp(tj - m)
                        + np.exp(lambda_draw) * np.exp(0.5 * (ti + tj) - m)
                    )
                )

                loglik -= Nij * logZ   

        return loglik


    def log_gradient(self, theta: np.array)-> np.array:

        """
        Compute and return minus the gradient of the log-likelihood of the model for a vector of strengths theta
        (We compute -gradient instead of +gradient to conduct a gradient descent and not a gradient ascent)

        Parameters
        ----------
        theta : np.array
            Vector of strengths
        
        Returns
        -------
        grad : np.array
            Value of the gradient of the log-likelihood for the given X and theta
        """

        W = self.W
        D = self.D
        lambda_draw = self.lambda_draw

        theta = np.asarray(theta)
        n = len(theta)

        grad = np.zeros(n, dtype=float)

        for k in range(n):
            for j in range(n):
                if j==k:
                    continue

                tk = theta[k]
                tj = theta[j]

                # Observed term
                obs = W[k, j] + 0.5 * D[k, j]

                # Total number of matches
                Nkj = W[k, j] + W[j, k] + D[k, j]

                # Numerically stable computation of expected term
                m = max(tk, tj)

                exp_k = np.exp(tk - m)
                exp_j = np.exp(tj - m)
                exp_d = np.exp(0.5 * (tk + tj) - m)

                Z = exp_k + exp_j + np.exp(lambda_draw) * exp_d

                # Expected contribution of player k
                Ek = (exp_k + 0.5 * np.exp(lambda_draw) * exp_d) / Z

                # -gradient accumulation
                grad[k] -= obs - Nkj * Ek

        return grad


    def fit(self, train_data : dict) -> BradleyTerry:

        """
        Fit the model to the data

        Parameters
        ----------
        train_data : dict
            Processed training data with Teams, Victory Matrix, Draw Matrix
            {'Teams': list of team names,
             'Victory Matrix': np.array of shape (n_teams, n_teams) with number of victories,
             'Draw Matrix': np.array of shape (n_teams, n_teams) with number of draws}

        Returns
        -------
        self : object
        """

        self.theta = np.zeros(train_data['Victory Matrix'].shape[0])
        self.W = train_data['Victory Matrix']
        self.D = train_data['Draw Matrix']

        self.teams = train_data['Teams']
        self.teams_index = {team:i for i, team in enumerate(self.teams)}


        # Gradient descent
        Optimizer = GradientDescent(learning_rate=self.learning_rate,
                                    n_iterations=self.n_iterations,
                                    tolerance=self.tolerance)
        
        self.theta = Optimizer.optimize(gradient=self.log_gradient,
                                        starting_point= self.theta)
        
        self.theta -= np.mean(self.theta)

        print("Model fitted successfully !")

        return self
    
    
    def predict_strength(self) -> pd.DataFrame:

        """
        Return the strengths of each team as a dataframe

        Returns
        -------
        strengths : pd.DataFrame
            DataFrame with columns 'Team' and 'Strength'
        """

        strengths = pd.DataFrame({
            'Team': self.teams,
            'Strength': [float(value) for value in self.theta]
        })

        return strengths
    
    
    
    def predict_proba(self, team1: str, team2: str) -> dict:

        """
        Predict the probability of each outcome between two teams
        
        Parameters
        ----------
        team1 : str
            Name of first team
        team2 : str
            Name of second team
            
        Returns
        -------
        probs : dict
            Dictionary with keys f'{team1}_win', 'draw', f'{team2}_win'
        """

        i = self.teams_index[team1]
        j = self.teams_index[team2]
        
        ti = self.theta[i]
        tj = self.theta[j]
        
        # Compute probabilities
        m = max(ti, tj)
        exp_i = np.exp(ti - m)
        exp_j = np.exp(tj - m)
        exp_d = np.exp(0.5 * (ti + tj) - m)
        
        Z = exp_i + exp_j + np.exp(self.lambda_draw) * exp_d
        
        return {
            f'{team1}_win': exp_i / Z,
            'draw': np.exp(self.lambda_draw) * exp_d / Z,
            f'{team2}_win': exp_j / Z
        }
    
    
    def simulate_season(self, list_teams: list) -> pd.DataFrame:
        
        """
        Simulate the results of a season using the estimated probabilities from the model.
        
        Parameters
        ----------
        list_teams : list
            List of teams to include in the simulation
            
        Returns
        -------
        df : pd.DataFrame
            DataFrame with columns 'HomeTeam', 'AwayTeam', 'P_Home', 'P_Draw', 'P_Away', 'Result'
        """
        
        matches = []
        
        # Generate all unique pairs
        for i in range(len(list_teams)):
            for j in range(i+1, len(list_teams)):
                team1 = list_teams[i]
                team2 = list_teams[j]
                
                # Match 1: team1 home, team2 away
                probs1 = self.predict_proba(team1, team2)
                p_home = probs1[f'{team1}_win']
                p_draw = probs1['draw']
                p_away = probs1[f'{team2}_win']
                
                # Determine result
                if p_home > p_draw and p_home > p_away:
                    result = 'H'
                elif p_draw > p_away:
                    result = 'D'
                else:
                    result = 'A'
                
                matches.append({
                    'HomeTeam': team1,
                    'AwayTeam': team2,
                    'P_Home': p_home,
                    'P_Draw': p_draw,
                    'P_Away': p_away,
                    'Result': result
                })
                
                # Match 2: team2 home, team1 away
                probs2 = self.predict_proba(team2, team1)
                p_home = probs2[f'{team2}_win']
                p_draw = probs2['draw']
                p_away = probs2[f'{team1}_win']
                
                # Determine result
                if p_home > p_draw and p_home > p_away:
                    result = 'H'
                elif p_draw > p_away:
                    result = 'D'
                else:
                    result = 'A'
                
                matches.append({
                    'HomeTeam': team2,
                    'AwayTeam': team1,
                    'P_Home': p_home,
                    'P_Draw': p_draw,
                    'P_Away': p_away,
                    'Result': result
                })
        
        df = pd.DataFrame(matches)
        return df


