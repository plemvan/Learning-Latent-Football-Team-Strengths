#========== Test/test_BT_classical.py ==========#

## Imports
import numpy as np
import pandas as pd
from typing import Literal
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr

#===============================================#


class BT_evaluate_strengths:
    """Class to evaluate BT classical strengths"""

    def __init__(self, BT_results : pd.DataFrame, NBTR_results : pd.DataFrame, test_df : pd.DataFrame):
        
        # Results
        self.BT_results = BT_results
        self.NBTR_results = NBTR_results

        # Will be used later to check consistency
        # if set(self.BT_results['Team']) != set(self.NBTR_results['Team']):
            # raise ValueError("Team index in BT_results and NBTR_results do not match.")

        self.results = self.BT_results.merge(self.NBTR_results, on='Team', how='inner')

        self.results = self.results.rename(columns={self.results.columns[0]: 'Team',
                                                    self.results.columns[1]: 'BT_strength',
                                                    self.results.columns[2]: 'NBTR_strength'})
 
        # Test data
        self.test_df = test_df

        # Hyper parameters for evaluation
        self.n_bootstrap = 1000
        self.confidence_level = 0.95

        # Performance metrics
        self.performance_metrics : pd.DataFrame = None

        return
    

    def plot_strengths(self, model : Literal['BT', 'NBTR']):
        
        """
        Docstring for plot_strengths
        
        :param self: Description
        :param model: Description
        :type model: Literal['BT', 'NBTR']
        """

        if model == 'BT':

            # sort by BT strengths
            df = self.results.sort_values(by='BT_strength', ascending=True)
            strengths = df['BT_strength']
            title = "Bradley-Terry Strengths"

        elif model == 'NBTR':

            # sort by NBTR strengths
            df = self.results.sort_values(by='NBTR_strength', ascending=True)
            strengths = df['NBTR_strength']
            title = "Neural Bradley-Terry Strengths"
            
        else:
            raise ValueError("Model must be either 'BT' or 'NBTR'.")

        plt.figure(figsize=(10, 6))
        plt.barh(df['Team'], strengths)
        plt.gca().invert_yaxis()
        plt.xlabel('Teams')
        plt.ylabel('Strength')
        plt.title(title)
        plt.show()

        return


    def pearson_correlation(self) -> dict:

        """
        Calculate Pearson correlation coefficient between BT and NBTR strengths
    
        Returns
        -------
        result : dict
            Dictionary containing:
            - 'correlation': Pearson correlation coefficient
            - 'pvalue': Two-tailed p-value for testing non-correlation
            - 'ci_lower': Lower bound of the confidence interval
            - 'ci_upper': Upper bound of the confidence interval
        """

        # Original correlation
        corr, pvalue = pearsonr(self.results['BT_strength'], self.results['NBTR_strength'])

        # Bootstrap for confidence intervals
        n = len(self.results)
        bootstrapped_corrs = []

        for _ in range(self.n_bootstrap):
            indices = np.random.choice(n, size=n, replace=True)
            sample = self.results.iloc[indices]

            # Calculate correlation for the bootstrap sample
            corr_boot, _ = pearsonr(sample['BT_strength'], sample['NBTR_strength'])
            bootstrapped_corrs.append(corr_boot)
        
        # Calculate confidence intervals
        alpha = 1 - self.confidence_level
        ci_lower = np.percentile(bootstrapped_corrs, 100 * (alpha / 2))
        ci_upper = np.percentile(bootstrapped_corrs, 100 * (1 - alpha / 2))

        return {
            'correlation': corr,
            'pvalue': pvalue,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper
        }


    def spearman_correlation(self) -> dict:

        """
        Calculate Spearman correlation coefficient between BT and NBTR strengths
    
        Returns
        -------
        result : dict
            Dictionary containing:
            - 'correlation': Spearman rank correlation coefficient
            - 'pvalue': Two-tailed p-value for testing non-correlation
            - 'ci_lower': Lower bound of the confidence interval
            - 'ci_upper': Upper bound of the confidence interval
        """

        # Original correlation
        corr, pvalue = spearmanr(self.results['BT_strength'], self.results['NBTR_strength'])

        # Bootstrap for confidence intervals
        n = len(self.results)
        bootstrapped_corrs = []

        for _ in range(self.n_bootstrap):
            indices = np.random.choice(n, size=n, replace=True)
            sample = self.results.iloc[indices]

            # Calculate correlation for the bootstrap sample
            corr_boot, _ = spearmanr(sample['BT_strength'], sample['NBTR_strength'])
            bootstrapped_corrs.append(corr_boot)

        # Calculate confidence intervals
        alpha = 1 - self.confidence_level
        ci_lower = np.percentile(bootstrapped_corrs, 100 * (alpha / 2))
        ci_upper = np.percentile(bootstrapped_corrs, 100 * (1 - alpha / 2))

        return {
            'correlation': corr,
            'pvalue': pvalue,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper
        }





    def test(self):

        return self.results