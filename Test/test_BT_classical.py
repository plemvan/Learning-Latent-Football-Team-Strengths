#========== Test/test_BT_classical.py ==========#

## Imports
import numpy as np
import pandas as pd
from scipy.special import expit
from scipy.stats import spearmanr, kendalltau
from BradleyTerry_classical.BT_classical import BradleyTerry


#===============================================#

class BT_classical_TEST:
    """Class to test the classical model"""

    def __init__(self, model : BradleyTerry):
        
        # Model
        self.model = model

        self.matrix = self.model.X
        self.teams = self.model.teams

        # Parameters
        self.learning_rate = self.model.learning_rate
        self.n_iterations = self.model.n_iterations
        self.tolerance = self.model.tolerance

        # Outputs
        self.theta = self.model.theta
        self.loglik_opt = self.model.loglikelihood(self.theta)
        self.grad_opt = self.model.log_gradient(self.theta)


        self.line = "="*60

        return
    

    def check_CV(self):

        """Check the convergence of the gradient descent"""

        grad_norm = np.linalg.norm(self.grad_opt)

        print("\n")
        print(self.line)
        print("CONVERGENCE CHECK")
        print(self.line)

        print(f"Gradient Norm: {grad_norm:.6e}")
        print(f"Max gradient component (abs. value): {np.max(np.abs(self.grad_opt)):.6e}")
        print(f"Tolerance: {self.tolerance}")

        if grad_norm < self.tolerance:
            print("You reached CV")
        elif grad_norm < 1e-3:
            print("Convergence OK but not perfect")
        else:
            print("NO CV - BAD :(")

        return


    def check_loglik(self):
        """Check the value of loglikelihood"""

        print("\n")
        print(self.line)
        print("LOG-LIKELIHOOD CHECK")
        print(self.line)

        print(f"Log-likelihood at the optimum: {self.loglik_opt:.4f}")

        # Compare with random points
        n_rd = 20
        rd_loglik = []

        for _ in range(n_rd):
            rd_theta = np.random.randn(len(self.teams))
            rd_theta -= np.mean(rd_theta)
            rd_loglik.append(self.model.loglikelihood(rd_theta))

        mean_rd = np.mean(rd_loglik)
        max_rd = np.max(rd_loglik)
        min_rd = np.min(rd_loglik)

        print(f"\nRandom points (n={n_rd}):")
        print(f"    - Max: {max_rd:.4f}")
        print(f"    - Mean: {mean_rd:.4f}")
        print(f"    - Min: {min_rd:.4f}")

        print(f"\nDifference between optimum loglik - best random loglik: {self.loglik_opt - max_rd:.4f}")

        if self.loglik_opt > max_rd:
            print("The optimum is better than any random point !")
        else:
            print("A random point is better - the gradient descent did not converge properly")

        
        # Compare with small perturbations
        n_pert = 200
        pert_sigma = 0.1
        pert_loglik = []

        for _ in range(n_pert):
            pert_theta = self.theta + pert_sigma*np.random.randn(len(self.teams))
            pert_theta -= np.mean(pert_theta)
            pert_loglik.append(self.model.loglikelihood(pert_theta))
        
        mean_pert = np.mean(pert_loglik)
        max_pert = np.max(pert_loglik)
        min_pert = np.min(pert_loglik)

        print(f"\nPerturbated optimum (n={n_pert}):")
        print(f"    - Max: {max_pert:.4f}")
        print(f"    - Mean: {mean_pert:.4f}")
        print(f"    - Min: {min_pert:.4f}")

        print(f"\nDifference between optimum loglik - best perturbated loglik: {self.loglik_opt - max_pert:.4f}")

        if self.loglik_opt > max_rd:
            print("The optimum is better than any perturbated point !")
        else:
            print("A perturbated point is better - the gradient descent did not converge properly")

        return
    

    def check_pred_quality(self):
        """Check the quality of predictions"""

        print("\n")
        print(self.line)
        print("PREDICTIONS QUALITY CHECK")
        print(self.line)

        # Predictions and observations
        predictions = []
        observations = []
        match_counts = []

        for i in range(len(self.teams)):
            for j in range(i+1, len(self.teams)):
                total_matches = self.matrix[i,j] + self.matrix[j,i]

                if total_matches > 0:
                    # Predicted proba for i beats j
                    p_pred = expit(self.theta[i] - self.theta[j])

                    # Observed freq
                    p_obs = self.matrix[i,j] / total_matches

                    predictions.append(p_pred)
                    observations.append(p_obs)
                    match_counts.append(total_matches)

        predictions = np.array(predictions)
        observations = np.array(observations)
        match_counts = np.array(match_counts)

        # Metrics
        mae = np.mean(np.abs(predictions - observations))
        rmse = np.sqrt(np.mean((predictions - observations)**2))
        correlation = np.corrcoef(predictions, observations)[0,1]

        print(f"Number of analyzed pairs: {len(predictions)}")
        print(f"\nMetrics:")
        print(f"    - MAE (Mean Absolute Error): {mae:.4f}")
        print(f"    - RMSE (R Mean Squared Error): {rmse:.4f}")
        print(f"\nCorrelation between pred/obs: {correlation:.4f}")

        return
    

    def check_rating_coherence(self):
        """Check coherence of ratings"""

        print("\n")
        print(self.line)
        print("RATING COHERENCE CHECK")
        print(self.line)

        # Empirical winrate
        empirical_winrate = []
        total_match_per_team = []

        for i in range(len(self.teams)):
            total_wins = np.sum(self.matrix[i,:])
            total_matches = np.sum(self.matrix[i,:] + self.matrix[:,i])

            if total_matches > 0:
                winrate = total_wins / total_matches
            else:
                winrate = 0

            empirical_winrate.append(winrate)
            total_match_per_team.append(total_matches)
        
        empirical_winrate = np.array(empirical_winrate)

        # Rank Correlation
        spearman_corr, spearman_pval = spearmanr(self.theta, empirical_winrate)
        kendall_corr, kendall_pval = kendalltau(self.theta, empirical_winrate)

        print(f"Spearman Correlation: {spearman_corr:.4f} (p-value: {spearman_pval:.2e})")
        print(f"Kendall Tau: {kendall_corr:.4f} (p-value: {kendall_pval:.2e})")

        # Display rankings
        print("\n")
        print(self.line)
        print("RANKING COMPARISON")
        print(self.line)

        # By theta
        rank_theta = np.argsort(-self.theta)
        print("\nRanking by theta (Bradley-Terry):")
        print(f"{'Rank':<6} {'Team':<25} {'theta':<10} {'Winrate'}")
        print("-"*60)
        for idx, i in enumerate(rank_theta[:10]):
            print(f"{idx+1:<6} {self.teams[i]:<25} {self.theta[i]:>6.3f}    {empirical_winrate[i]:.3f}")
        
        # By winrate
        rank_winrate = np.argsort(-empirical_winrate)
        print("\nRanking by winrate:")
        print(f"{'Rank':<6} {'Team':<25} {'Winrate':<15} {'theta'}")
        print("-"*60)
        for idx, i in enumerate(rank_winrate[:10]):
            print(f"{idx:<6} {self.teams[i]:<25} {empirical_winrate[i]:.3f}           {self.theta[i]:>6.3f}")
        
        return
    

    def run_all(self):

        self.check_CV()
        self.check_loglik()
        self.check_pred_quality()
        self.check_rating_coherence()

        return


