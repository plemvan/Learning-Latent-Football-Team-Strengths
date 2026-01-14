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
    

    def calculate_performance_metrics(self) -> pd.DataFrame:

        """
        Create and cache a DataFrame of performance metrics for BT and NBTR strengths
        
        :param self: Description
        :return: Description
        :rtype: DataFrame
        """

        df = self.test_df

        # All teams
        teams = pd.unique(self.test_df[['HomeTeam','AwayTeam']].values.ravel())
        
        # Matches played
        games = (
            df['HomeTeam'].value_counts()
            .add(df['AwayTeam'].value_counts(), fill_value=0)
        )

        # Wins
        home_wins = df[df['FTR'] == 'H']['HomeTeam'].value_counts()
        away_wins = df[df['FTR'] == 'A']['AwayTeam'].value_counts()
        wins = home_wins.add(away_wins, fill_value=0)

        # Draws
        home_draws = df[df['FTR'] == 'D']['HomeTeam'].value_counts()
        away_draws = df[df['FTR'] == 'D']['AwayTeam'].value_counts()
        draws = home_draws.add(away_draws, fill_value=0)

        # Points
        points = 3 * wins + draws

        # Assemble DataFrame
        scores_df = pd.DataFrame({
            "Team": teams,
            "Points": points.reindex(teams).fillna(0),
            "WinRate": (wins / games).reindex(teams).fillna(0),
            "WinRateAdj": ((wins + 0.5 * draws) / games).reindex(teams).fillna(0)
        })

        scores_df = scores_df.sort_values("Points", ascending=False).reset_index(drop=True)

        self.performance_metrics = scores_df

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
        plt.xlabel('Teams')
        plt.ylabel('Strength')
        plt.title(title)
        plt.show()

        return


    def plot_scatter_strengths(self):
        
        """
        Plot a scatter plot of BT strengths vs NBTR strengths with the y=x line.
        """
        
        plt.figure(figsize=(8, 6))
        plt.scatter(self.results['BT_strength'], self.results['NBTR_strength'], alpha=0.7)
        plt.plot([self.results['BT_strength'].min(), self.results['BT_strength'].max()], 
                 [self.results['BT_strength'].min(), self.results['BT_strength'].max()], 
                 color='red', linestyle='--', label='y=x')
        
        # Add team labels
        for _, row in self.results.iterrows():
            plt.annotate(row['Team'], 
                         (row['BT_strength'], row['NBTR_strength']), 
                         textcoords="offset points", 
                         xytext=(5,5), 
                         ha='left', 
                         fontsize=8, 
                         alpha=0.8)
        
        plt.xlabel('BT Strengths')
        plt.ylabel('NBTR Strengths')
        plt.title('Scatter Plot of BT vs NBTR Strengths')
        plt.legend()
        plt.grid(True)
        plt.show()

        return


    def plot_scatter_ranks(self):
        
        """
        Plot a scatter plot of BT ranks vs NBTR ranks with the y=x line.
        Ranks are based on descending strengths (higher strength = lower rank).
        """
        
        # Use a copy to avoid modifying self.results
        df = self.results.copy()
        df['BT_rank'] = df['BT_strength'].rank(ascending=False, method='dense')
        df['NBTR_rank'] = df['NBTR_strength'].rank(ascending=False, method='dense')
        
        plt.figure(figsize=(8, 6))
        plt.scatter(df['BT_rank'], df['NBTR_rank'], alpha=0.7)
        plt.plot([1, df['BT_rank'].max()], [1, df['BT_rank'].max()], 
                 color='red', linestyle='--', label='y=x')
        
        # Add team labels
        for _, row in df.iterrows():
            plt.annotate(row['Team'], 
                         (row['BT_rank'], row['NBTR_rank']), 
                         textcoords="offset points", 
                         xytext=(5,5), 
                         ha='left', 
                         fontsize=8, 
                         alpha=0.8)
        
        plt.xlabel('BT Ranks')
        plt.ylabel('NBTR Ranks')
        plt.title('Scatter Plot of BT vs NBTR Ranks')
        plt.legend()
        plt.grid(True)
        
        # Set integer ticks
        max_rank = max(df['BT_rank'].max(), df['NBTR_rank'].max())
        step = 2 # max(1, int(max_rank // 10))  # Adjust step for readability
        ticks = list(range(1, int(max_rank) + 1, step))
        plt.xticks(ticks)
        plt.yticks(ticks)
        
        plt.gca().invert_yaxis()  # Invert y-axis so rank 1 is at the top
        plt.gca().invert_xaxis()  # Invert x-axis so rank 1 is at the right
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
    

    def _spearman_with_bootstrap(self, x, y):

        """
        Docstring for _spearman_with_bootstrap
        
        :param self: Description
        :param x: Description
        :param y: Description
        """

        # Similar to spearman_correlation but for two arbitrary arrays

        # Original correlation
        corr, pvalue = spearmanr(x, y)

        # Bootstrap for confidence intervals
        n = len(x)
        bootstrapped_corrs = []
        for _ in range(self.n_bootstrap):
            indices = np.random.choice(n, size=n, replace=True)
            x_sample = x[indices]
            y_sample = y[indices]

            # Calculate correlation for the bootstrap sample
            corr_boot, _ = spearmanr(x_sample, y_sample)
            bootstrapped_corrs.append(corr_boot)

            # Calculate correlation for the bootstrap sample
            corr_boot, _ = spearmanr(x_sample, y_sample)
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


    def results_correlation(self, model : Literal['BT', 'NBTR'], metric : Literal['points','rank', 'winrate', 'adjusted_winrate']) -> dict:

        """
        Docstring for compare_with_results
        
        :param self: Description
        :param model: Description
        :type model: Literal['BT', 'NBTR']
        :param metric: Description
        :type metric: Literal['points', 'rank', 'winrate', 'adjusted_winrate']
        :return: Description
        :rtype: dict
        """

        # Performances metrics
        if self.performance_metrics is None:
            self.calculate_performance_metrics()
        perf_df = self.performance_metrics

        # Merge with self.results
        metric_name_dict = {'points' : 'Points',
                            'rank' : 'Points',
                            'winrate' : 'WinRate',
                            'adjusted_winrate' : 'WinRateAdj'}
        
        compare_col = metric_name_dict[metric]

        merged_df = self.results.merge(perf_df[['Team', compare_col]], on='Team', how='inner')

        # Select strengths based on model
        strength_col = f"{model}_strength"

        # Calculate Spearman correlation
        result = self._spearman_with_bootstrap(merged_df[strength_col], merged_df[compare_col])

        return result


class BT_evaluate_probas:
    """Class to evaluate BT classical strengths"""  

    def __init__(self, BT_results : pd.DataFrame, NBTR_results : pd.DataFrame, test_df : pd.DataFrame):

        # Results
        self.BT_results = BT_results
        self.NBTR_results = NBTR_results

        # Test data
        self.test_df = test_df

        # Final results
        self.results = self.test_df[['HomeTeam', 'AwayTeam', 'FTR']].merge(
            self.BT_results, 
            on=['HomeTeam', 'AwayTeam'], 
            how='inner', 
            suffixes=('', '_BT')
        )
        self.results = self.results.merge(
            self.NBTR_results, 
            on=['HomeTeam', 'AwayTeam'], 
            how='inner', 
            suffixes=('', '_NBTR')
        )
        
        return

    def logloss(self, model : Literal['BT','NBTR']) -> float:

        """
        Calculate the log-loss for the specified model using self.results.
        
        Parameters
        ----------
        model : str
            'BT' for Bradley-Terry classical, 'NBTR' for Neural Bradley-Terry
            
        Returns
        -------
        float
            The average log-loss over all matches
        """
        
        if model == 'BT':
            p_home_col = 'P_Home'
            p_draw_col = 'P_Draw'
            p_away_col = 'P_Away'
        elif model == 'NBTR':
            p_home_col = 'P_Home_NBTR'
            p_draw_col = 'P_Draw_NBTR'
            p_away_col = 'P_Away_NBTR'
        else:
            raise ValueError("Model must be 'BT' or 'NBTR'")
        
        logloss_sum = 0.0
        n_matches = len(self.results)
        
        for _, row in self.results.iterrows():
            ftr = row['FTR']
            if ftr == 'H':
                p = row[p_home_col]
            elif ftr == 'D':
                p = row[p_draw_col]
            elif ftr == 'A':
                p = row[p_away_col]
            else:
                raise ValueError(f"Unknown FTR value: {ftr}")
            
            logloss_sum += -np.log(p)
        
        return logloss_sum / n_matches
    

    def accuracy(self, model : Literal['BT','NBTR']) -> float:

        """
        Calculate the accuracy for the specified model.
        
        Parameters
        ----------
        model : str
            'BT' or 'NBTR'
            
        Returns
        -------
        float
            Accuracy as proportion of correct predictions
        """
        
        if model == 'BT':
            result_col = 'Result'
        elif model == 'NBTR':
            result_col = 'Result_NBTR'
        else:
            raise ValueError("Model must be 'BT' or 'NBTR'")
        
        correct = (self.results[result_col] == self.results['FTR']).sum()
        return correct / len(self.results)
    
    
    def recall(self, model : Literal['BT','NBTR']) -> dict:
        
        """
        Calculate the recall for each class ('H', 'D', 'A') for the specified model.
        
        Parameters
        ----------
        model : str
            'BT' or 'NBTR'
            
        Returns
        -------
        dict
            Dictionary with recall for each class
        """
        
        if model == 'BT':
            result_col = 'Result'
        elif model == 'NBTR':
            result_col = 'Result_NBTR'
        else:
            raise ValueError("Model must be 'BT' or 'NBTR'")
        
        recalls = {}
        for cls in ['H', 'D', 'A']:
            tp = ((self.results[result_col] == cls) & (self.results['FTR'] == cls)).sum()
            fn = ((self.results[result_col] != cls) & (self.results['FTR'] == cls)).sum()
            recalls[cls] = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        
        return recalls
    
    
    def precision(self, model : Literal['BT','NBTR']) -> dict:
        
        """
        Calculate the precision for each class ('H', 'D', 'A') for the specified model.
        
        Parameters
        ----------
        model : str
            'BT' or 'NBTR'
            
        Returns
        -------
        dict
            Dictionary with precision for each class
        """
        
        if model == 'BT':
            result_col = 'Result'
        elif model == 'NBTR':
            result_col = 'Result_NBTR'
        else:
            raise ValueError("Model must be 'BT' or 'NBTR'")
        
        precisions = {}
        for cls in ['H', 'D', 'A']:
            tp = ((self.results[result_col] == cls) & (self.results['FTR'] == cls)).sum()
            fp = ((self.results[result_col] == cls) & (self.results['FTR'] != cls)).sum()
            precisions[cls] = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        
        return precisions
    
    
    def brier_score(self, model : Literal['BT','NBTR']) -> float:
        
        """
        Calculate the Brier score for the specified model.
        
        Parameters
        ----------
        model : str
            'BT' or 'NBTR'
            
        Returns
        -------
        float
            The Brier score
        """
        
        if model == 'BT':
            p_home_col = 'P_Home'
            p_draw_col = 'P_Draw'
            p_away_col = 'P_Away'
        elif model == 'NBTR':
            p_home_col = 'P_Home_NBTR'
            p_draw_col = 'P_Draw_NBTR'
            p_away_col = 'P_Away_NBTR'
        else:
            raise ValueError("Model must be 'BT' or 'NBTR'")
        
        brier_sum = 0.0
        n_matches = len(self.results)
        
        for _, row in self.results.iterrows():
            ftr = row['FTR']
            p_home = row[p_home_col]
            p_draw = row[p_draw_col]
            p_away = row[p_away_col]
            
            if ftr == 'H':
                o_home, o_draw, o_away = 1, 0, 0
            elif ftr == 'D':
                o_home, o_draw, o_away = 0, 1, 0
            elif ftr == 'A':
                o_home, o_draw, o_away = 0, 0, 1
            else:
                raise ValueError(f"Unknown FTR value: {ftr}")
            
            brier_sum += (p_home - o_home)**2 + (p_draw - o_draw)**2 + (p_away - o_away)**2
        
        return brier_sum / n_matches
    
    
    def reliability_curve_draw(self, model : Literal['BT','NBTR'], n_bins : int = 10, plot : bool = False) -> pd.DataFrame:
        
        """
        Calculate the reliability curve data for the 'Draw' class.
        
        Parameters
        ----------
        model : str
            'BT' or 'NBTR'
        n_bins : int, default 10
            Number of bins for grouping probabilities
        plot : bool, default False
            If True, plot the reliability curve
            
        Returns
        -------
        pd.DataFrame
            DataFrame with columns: 'bin_center', 'mean_pred_prob', 'observed_freq', 'bin_size'
        """
        
        if model == 'BT':
            p_draw_col = 'P_Draw'
        elif model == 'NBTR':
            p_draw_col = 'P_Draw_NBTR'
        else:
            raise ValueError("Model must be 'BT' or 'NBTR'")
        
        # Create bins
        bins = np.linspace(0, 1, n_bins + 1)
        bin_labels = [(bins[i] + bins[i+1]) / 2 for i in range(n_bins)]  # bin centers
        
        # Assign bins to data
        self.results = self.results.copy()
        self.results['prob_bin'] = pd.cut(self.results[p_draw_col], bins=bins, labels=bin_labels, include_lowest=True)
        
        # Group by bin and calculate metrics
        grouped = self.results.groupby('prob_bin', observed=False).agg(
            mean_pred_prob=(p_draw_col, 'mean'),
            observed_freq=('FTR', lambda x: (x == 'D').mean()),
            bin_size=('FTR', 'size')
        ).reset_index()
        
        # Rename prob_bin to bin_center
        grouped = grouped.rename(columns={'prob_bin': 'bin_center'})
        
        if plot:
            plt.figure(figsize=(8, 6))
            plt.plot(grouped['mean_pred_prob'], grouped['observed_freq'], marker='o', label='Observed')
            plt.plot([0, 1], [0, 1], linestyle='--', color='red', label='Perfect calibration')
            plt.xlabel('Mean Predicted Probability')
            plt.ylabel('Observed Frequency')
            plt.title(f'Reliability Curve for Draw ({model})')
            plt.legend()
            plt.grid(True)
            plt.show()
        
        return grouped




