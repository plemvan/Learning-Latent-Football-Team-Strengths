#========== Test/test_BT_classical.py ==========#

## Imports
import numpy as np
from scipy.special import expit
from scipy.stats import spearmanr, kendalltau
from BradleyTerry_classical.BT_classical import BradleyTerry


#===============================================#

class BT_classical_TEST:
    """Class to test the classical model"""

    def __init__(self, model : BradleyTerry):
        
        # Model
        self.model = model

        self.lambda_draw = self.model.lambda_draw

        self.W_matrix = self.model.W
        self.D_matrix = self.model.D
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

        W = self.W_matrix
        D = self.D_matrix
        lambda_draw = self.lambda_draw

        # Predictions and observations for 3 outcomes
        predictions_win = []
        predictions_draw = []
        predictions_loss = []

        observations_win =[]
        observations_draw = []
        observations_loss = []

        match_counts = []

        for team1 in self.teams:
            for team2 in self.teams:
                if team1==team2:
                    continue

                i = self.model.teams_index[team1]
                j = self.model.teams_index[team2]

                total_matches = W[i,j] + W[j,i] + D[i,j]

                if total_matches >0 :

                    probas = self.model.predict_proba(team1, team2)

                    p_i_wins = probas[f'{team1}_win']
                    p_draw = probas['draw']
                    p_j_wins = probas[f'{team2}_win']

                    # Observed frequencies
                    freq_i_wins = W[i,j] / total_matches
                    freq_draw = D[i,j] / total_matches
                    freq_j_wins = W[j,i] / total_matches

                    predictions_win.append(p_i_wins)
                    predictions_draw.append(p_draw)
                    predictions_loss.append(p_j_wins)
                    
                    observations_win.append(freq_i_wins)
                    observations_draw.append(freq_draw)
                    observations_loss.append(freq_j_wins)
                    
                    match_counts.append(total_matches)
        
        # Convert to arrays
        predictions_win = np.array(predictions_win)
        predictions_draw = np.array(predictions_draw)
        predictions_loss = np.array(predictions_loss)
        
        observations_win = np.array(observations_win)
        observations_draw = np.array(observations_draw)
        observations_loss = np.array(observations_loss)
        
        match_counts = np.array(match_counts)

        # Global metrics (all 3 outcomes combined)
        all_predictions = np.concatenate([predictions_win, predictions_draw, predictions_loss])
        all_observations = np.concatenate([observations_win, observations_draw, observations_loss])
        
        mae_global = np.mean(np.abs(all_predictions - all_observations))
        rmse_global = np.sqrt(np.mean((all_predictions - all_observations)**2))
        correlation_global = np.corrcoef(all_predictions, all_observations)[0,1]

        # Metrics per outcome
        mae_win = np.mean(np.abs(predictions_win - observations_win))
        mae_draw = np.mean(np.abs(predictions_draw - observations_draw))
        mae_loss = np.mean(np.abs(predictions_loss - observations_loss))
        
        corr_win = np.corrcoef(predictions_win, observations_win)[0,1]
        corr_draw = np.corrcoef(predictions_draw, observations_draw)[0,1]
        corr_loss = np.corrcoef(predictions_loss, observations_loss)[0,1]

        print(f"Number of analyzed pairs: {len(predictions_win)}")
        print(f"Total outcomes analyzed: {3 * len(predictions_win)}")
        
        print(f"\n--- Global Metrics (all 3 outcomes) ---")
        print(f"    - MAE (Mean Absolute Error): {mae_global:.4f}")
        print(f"    - RMSE (Root Mean Squared Error): {rmse_global:.4f}")
        print(f"    - Correlation pred/obs: {correlation_global:.4f}")
        
        print(f"\n--- Metrics per Outcome ---")
        print(f"Win:  MAE = {mae_win:.4f}, Corr = {corr_win:.4f}")
        print(f"Draw: MAE = {mae_draw:.4f}, Corr = {corr_draw:.4f}")
        print(f"Loss: MAE = {mae_loss:.4f}, Corr = {corr_loss:.4f}")

        return
    

    def check_rating_coherence(self):
        """Check coherence of ratings"""

        print("\n")
        print(self.line)
        print("RATING COHERENCE CHECK")
        print(self.line)

        W = self.W_matrix
        D = self.D_matrix

        # Empirical performance (points per match)
        empirical_points = []
        empirical_winrate = []
        empirical_drawrate = []
        total_match_per_team = []

        for i in range(len(self.teams)):
            total_wins = np.sum(W[i,:])
            total_draws = np.sum(D[i,:])
            total_losses = np.sum(W[:,i])
            total_matches = total_wins + total_draws + total_losses

            if total_matches > 0:
                # Points system: win = 1, draw = 0.5, loss = 0
                points_per_match = (total_wins + 0.5 * total_draws) / total_matches
                winrate = total_wins / total_matches
                drawrate = total_draws / total_matches
            else:
                points_per_match = 0
                winrate = 0
                drawrate = 0

            empirical_points.append(points_per_match)
            empirical_winrate.append(winrate)
            empirical_drawrate.append(drawrate)
            total_match_per_team.append(total_matches)
        
        empirical_points = np.array(empirical_points)
        empirical_winrate = np.array(empirical_winrate)
        empirical_drawrate = np.array(empirical_drawrate)

        # Rank Correlation (using points per match)
        spearman_corr, spearman_pval = spearmanr(self.theta, empirical_points)
        kendall_corr, kendall_pval = kendalltau(self.theta, empirical_points)

        print(f"Spearman Correlation (theta vs points/match): {spearman_corr:.4f} (p-value: {spearman_pval:.2e})")
        print(f"Kendall Tau (theta vs points/match): {kendall_corr:.4f} (p-value: {kendall_pval:.2e})")

        # Display rankings
        print("\n")
        print(self.line)
        print("RANKING COMPARISON")
        print(self.line)

        # By theta
        rank_theta = np.argsort(-self.theta)
        print("\nTop 10 by theta (Bradley-Terry-Davidson):")
        print(f"{'Rank':<6} {'Team':<25} {'theta':<10} {'Pts/Match':<12} {'Win%':<8} {'Draw%'}")
        print("-"*80)
        for idx, i in enumerate(rank_theta[:10]):
            print(f"{idx+1:<6} {self.teams[i]:<25} {self.theta[i]:>6.3f}    "
                  f"{empirical_points[i]:>6.3f}      {empirical_winrate[i]:>5.1%}   {empirical_drawrate[i]:>5.1%}")
        
        # By points per match
        rank_points = np.argsort(-empirical_points)
        print("\nTop 10 by points per match:")
        print(f"{'Rank':<6} {'Team':<25} {'Pts/Match':<12} {'theta':<10} {'Win%':<8} {'Draw%'}")
        print("-"*80)
        for idx, i in enumerate(rank_points[:10]):
            print(f"{idx+1:<6} {self.teams[i]:<25} {empirical_points[i]:>6.3f}      "
                  f"{self.theta[i]:>6.3f}    {empirical_winrate[i]:>5.1%}   {empirical_drawrate[i]:>5.1%}")
        
        # Statistics on draws
        print("\n")
        print(self.line)
        print("DRAW STATISTICS")
        print(self.line)
        print(f"Lambda_draw parameter: {self.lambda_draw:.4f}")
        print(f"Mean draw rate across all teams: {np.mean(empirical_drawrate):.1%}")
        print(f"Std draw rate: {np.std(empirical_drawrate):.1%}")
        
        return
    

    def run_all(self):

        self.check_CV()
        self.check_loglik()
        self.check_pred_quality()
        self.check_rating_coherence()

        return


