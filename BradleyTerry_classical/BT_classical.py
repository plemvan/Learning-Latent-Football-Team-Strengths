#========== Classical Bradley-Terry Model ==========#

## Imports
import numpy as np
from scipy.special import expit
from .Gradient_descent import GradientDescent


#===================================================#

class BradleyTerry():

    """Class implementing the classical Bradley-Terry framework"""

    def __init__(self, lambda_draw: float, learning_rate : float = 0.01, n_iterations : int = 1000, tolerance : float = 1e-6):
        
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
        Compute and return the log-likelihood of the model for a vector of strengths theta and the matrix of results X

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
        Compute and return the -(gradient of the log-likelihood) of the model for a vector of strenghts theta and the matrix of results X
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

                # ----- Stable computation of expected term -----
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


    def fit(self, W : np.array, D: np.array, teams : list) -> BradleyTerry:

        """
        Fit the model to the data

        Parameters
        ----------
        W : np.array
            Matrix of victories. W[i,j] = nb of victories of i against j for all i != j
        D : np.array
            Matrix of draws (symmetric). D[i,j] = nb of draws between i and j for all i != j

        Returns
        -------
        self : object
        """

        self.theta = np.zeros(W.shape[0])
        self.W = W
        self.D = D

        self.teams = teams
        self.teams_index = {team:i for i, team in enumerate(teams)}


        # Gradient descent
        Optimizer = GradientDescent(learning_rate=self.learning_rate,
                                    n_iterations=self.n_iterations,
                                    tolerance=self.tolerance)
        
        self.theta = Optimizer.optimize(gradient=self.log_gradient,
                                        starting_point= self.theta)
        
        self.theta -= np.mean(self.theta)

        print("Model fitted successfully !")

        return self
    
    
    def predict_strength(self) -> np.array:

        """
        Return the strengths of each team

        Returns
        -------
        theta : np.array
            Array of strengths
        """
        
        return self.theta
    
    
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
            Dictionary with keys 'team1_win', 'draw', 'team2_win'
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

