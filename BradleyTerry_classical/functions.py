#========== Functions for BT Classical ==========#

## Imports
import numpy as np
import matplotlib.pyplot as plt
from .BT_classical import BradleyTerry

#================================================#


class BT_calibrate:

    """Class to calibrate BT classical"""

    def __init__(self, train_data : dict):

        """
        Calibrate BT classical model

        Parameters
        ----------
        train_data : dict
            Processed training data with Teams, Victory Matrix, Draw Matrix
            {'Teams': list of team names,
             'Victory Matrix': np.array of shape (n_teams, n_teams) with number of victories,
             'Draw Matrix': np.array of shape (n_teams, n_teams) with number of draws}
        """

        self.data = train_data
        
        self.W = self.data['Victory Matrix']
        self.D = self.data['Draw Matrix']
        self.teams = self.data['Teams']

        return
    
    def calibrate_lambda(self, lambda_grid : np.linspace):

        """
        Calibrate lambda via profile likelihood approach

        Parameters
        ----------
        lambda_grid : np.linspace
            Grid of lambda values to test
        """

        logliks = []

        for lambda_draw in lambda_grid:

            bt_model = BradleyTerry(lambda_draw=lambda_draw,
                                    learning_rate=0.01,
                                    n_iterations=1500)
            
            bt_model.fit(train_data=self.data)
            
            theta_hat = bt_model.predict_strength()['Strength'].tolist()
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