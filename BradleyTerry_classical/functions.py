#========== Functions for BT Classical ==========#

## Imports
import numpy as np
import matplotlib.pyplot as plt
from .BT_classical import BradleyTerry

#================================================#


class BT_calibrate:
    """Class to calibrate BT classical"""

    def __init__(self, W : np.array, D : np.array, teams : list):

        """
        Docstring for __init__
        
        :param self: Description
        :param W: Description
        :type W: np.array
        :param D: Description
        :type D: np.array
        """
        
        self.W = W
        self.D = D
        self.teams = teams

        return
    
    def calibrate_lambda(self, lambda_grid : np.linspace):

        """
        Docstring for calibrate_lambda
        
        :param self: Description
        :param lambda_grid: Description
        :type lambda_grid: np.linspace
        """

        logliks = []

        for lambda_draw in lambda_grid:

            bt_model = BradleyTerry(lambda_draw=lambda_draw,
                                    learning_rate=0.01,
                                    n_iterations=1500)
            
            bt_model.fit(W = self.W, D=self.D, teams=self.teams)
            
            theta_hat = list(bt_model.predict_strength().values())
            loglik = bt_model.loglikelihood(theta=theta_hat)
            logliks.append(loglik)

        lambda_opt = lambda_grid[np.argmax(logliks)]

        plt.plot(lambda_grid, logliks)
        plt.plot(lambda_opt, max(logliks), 'ro', label='Optimal lambda')
        plt.annotate(f'({lambda_opt:.2f}, {max(logliks):.2f})', xy=(lambda_opt, max(logliks)), xytext=(lambda_opt - 0.5, max(logliks) + 0.1), ha='center')
        plt.xlabel("lambda")
        plt.ylabel("Log-likelihood")
        plt.title("Profile Log-likelihood for lambda")
        plt.grid()
        plt.show()

        return