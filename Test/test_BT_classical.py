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

        # Check availability
        self.has_BT = self.BT_results is not None
        self.has_NBTR = self.NBTR_results is not None

        # Merge results
        if not self.has_BT and not self.has_NBTR:
            raise ValueError("At least one of BT_results or NBTR_results must be provided.")
        
        if self.has_BT and not self.has_NBTR:
            self.results = self.BT_results.copy()
            self.results = self.results.rename(columns={self.results.columns[0]: 'Team',
                                                    self.results.columns[1]: 'BT_strength'})
        elif not self.has_BT and self.has_NBTR:
            self.results = self.NBTR_results.copy()
            self.results = self.results.rename(columns={self.results.columns[0]: 'Team',
                                                    self.results.columns[1]: 'NBTR_strength'})
        else:
            self.results = self.BT_results.merge(self.NBTR_results, on='Team', how='right')
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
        Plot strengths of the model as horizontal bars

        Parameters
        ----------
        model : str
            'BT' for Bradley-Terry classical, 'NBTR' for Neural Bradley-Terry
        """

        if model == 'BT':
            if not self.has_BT:
                raise ValueError("BT results are not available.")
            # sort by BT strengths
            df = self.results.sort_values(by='BT_strength', ascending=True)
            strengths = df['BT_strength']
            title = "Bradley-Terry Strengths"

        elif model == 'NBTR':
            if not self.has_NBTR:
                raise ValueError("NBTR results are not available.")
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


    def plot_scatter_ranks(self):
        
        """
        Plot a scatter plot of BT ranks vs NBTR ranks with the y=x line.
        Ranks are based on descending strengths (higher strength = lower rank).
        """
        
        if not (self.has_BT and self.has_NBTR):
            raise ValueError("Both BT and NBTR results are required for this method.")
        
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
        step = 2
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

        if not (self.has_BT and self.has_NBTR):
            raise ValueError("Both BT and NBTR results are required for this method.")

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

        if not (self.has_BT and self.has_NBTR):
            raise ValueError("Both BT and NBTR results are required for this method.")

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
        Internal method to compute spearman correlation with bootstrap confidence intervals on arbitrary arrays
        
        Parameters
        ----------
        x : array-like
            First array of values
        y : array-like
            Second array of values

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
        Measure Spearman correlation between model strengths and performance metrics (points, rank, winrate, adjusted winrate)

        Parameters
        ----------
        model : str
            'BT' for Bradley-Terry classical, 'NBTR' for Neural Bradley-Terry
        metric : str
            'points', 'rank', 'winrate', or 'adjusted_winrate'
        
        Returns
        -------
        result : dict
            Dictionary containing:
            - 'correlation': Spearman rank correlation coefficient
            - 'pvalue': Two-tailed p-value for testing non-correlation
            - 'ci_lower': Lower bound of the confidence interval
            - 'ci_upper': Upper bound of the confidence interval
        """

        if model == 'BT' and not self.has_BT:
            raise ValueError("BT results are not available.")
        elif model == 'NBTR' and not self.has_NBTR:
            raise ValueError("NBTR results are not available.")

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
    

    ## Unused methods

    def plot_scatter_strengths(self):
        
        """
        Plot a scatter plot of BT strengths vs NBTR strengths with the y=x line.
        """
        
        if not (self.has_BT and self.has_NBTR):
            raise ValueError("Both BT and NBTR results are required for this method.")
        
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


class BT_evaluate_probas:
    """Class to evaluate BT classical strengths"""  

    def __init__(self, BT_results : pd.DataFrame, NBTR_results : pd.DataFrame, test_df : pd.DataFrame):

        # Results
        self.BT_results = BT_results
        self.NBTR_results = NBTR_results

        # Check availability
        self.has_BT = self.BT_results is not None
        self.has_NBTR = self.NBTR_results is not None

        if not (self.has_BT or self.has_NBTR):
            raise ValueError("At least one of BT_results or NBTR_results must be provided.")

        # Test data
        self.test_df = test_df

        # Final results
        base_df = self.test_df[['HomeTeam', 'AwayTeam', 'FTR']].copy()
        
        if self.has_BT:
            base_df = base_df.merge(
                self.BT_results, 
                on=['HomeTeam', 'AwayTeam'], 
                how='inner',
                suffixes=('', '_BT') if self.has_NBTR else ('', '')
            )
        
        if self.has_NBTR:
            base_df = base_df.merge(
                self.NBTR_results, 
                on=['HomeTeam', 'AwayTeam'], 
                how='inner', 
                suffixes=('', '_NBTR') if self.has_BT else ('', '')
            )
        
        self.results = base_df
        
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
            if not self.has_BT:
                raise ValueError("BT results are not available.")
            p_home_col = 'P_Home'
            p_draw_col = 'P_Draw'
            p_away_col = 'P_Away'
        elif model == 'NBTR':
            if not self.has_NBTR:
                raise ValueError("NBTR results are not available.")
            p_home_col = 'P_Home' if not self.has_BT else 'P_Home_NBTR'
            p_draw_col = 'P_Draw' if not self.has_BT else 'P_Draw_NBTR'
            p_away_col = 'P_Away' if not self.has_BT else 'P_Away_NBTR'
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
            'BT' for Bradley-Terry classical, 'NBTR' for Neural Bradley-Terry
            
        Returns
        -------
        accuracy : float
            Accuracy as proportion of correct predictions
        """
        
        if model == 'BT':
            if not self.has_BT:
                raise ValueError("BT results are not available.")
            result_col = 'Result'
        elif model == 'NBTR':
            if not self.has_NBTR:
                raise ValueError("NBTR results are not available.")
            result_col = 'Result' if not self.has_BT else 'Result_NBTR'
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
            'BT' for Bradley-Terry classical, 'NBTR' for Neural Bradley-Terry
            
        Returns
        -------
        recalls : dict
            Dictionary with recall for each class
        """
        
        if model == 'BT':
            if not self.has_BT:
                raise ValueError("BT results are not available.")
            result_col = 'Result'
        elif model == 'NBTR':
            if not self.has_NBTR:
                raise ValueError("NBTR results are not available.")
            result_col = 'Result' if not self.has_BT else 'Result_NBTR'
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
            'BT' for Bradley-Terry classical, 'NBTR' for Neural Bradley-Terry
            
        Returns
        -------
        precisions : dict
            Dictionary with precision for each class
        """
        
        if model == 'BT':
            if not self.has_BT:
                raise ValueError("BT results are not available.")
            result_col = 'Result'
        elif model == 'NBTR':
            if not self.has_NBTR:
                raise ValueError("NBTR results are not available.")
            result_col = 'Result' if not self.has_BT else 'Result_NBTR'
        else:
            raise ValueError("Model must be 'BT' or 'NBTR'")
        
        precisions = {}
        for cls in ['H', 'D', 'A']:
            tp = ((self.results[result_col] == cls) & (self.results['FTR'] == cls)).sum()
            fp = ((self.results[result_col] == cls) & (self.results['FTR'] != cls)).sum()
            precisions[cls] = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        
        return precisions


    def accuracy_per_team(self, model : Literal['BT','NBTR']):

        """
        Compute accuracy for every team in the DataFrame
        
        Parameters
        ----------
        model : str
            'BT' for Bradley-Terry classical, 'NBTR' for Neural Bradley-Terry
        """

        if model == 'BT':
            if not self.has_BT:
                raise ValueError("BT results are not available.")
            result_col = 'Result'
        elif model == 'NBTR':
            if not self.has_NBTR:
                raise ValueError("NBTR results are not available.")
            result_col = 'Result' if not self.has_BT else 'Result_NBTR'
        else:
            raise ValueError("Model must be 'BT' or 'NBTR'")
        
        # Unique teams
        teams = pd.unique(self.results[['HomeTeam','AwayTeam']].values.ravel())

        results = []

        for team in teams:

            df_team = self.results[(self.results['HomeTeam']==team)|(self.results['AwayTeam']==team)]

            if len(df_team)==0:
                continue

            accuracy = (df_team['FTR'] == df_team[result_col]).mean()

            results.append({
            'Team': team,
            'Accuracy': accuracy
            })

        acc_df = pd.DataFrame(results).sort_values('Accuracy', ascending=False).reset_index(drop=True)
        
        # ---- Plot ----
        plt.figure(figsize=(10, max(6, 0.35 * len(acc_df))))

        # Default colors
        colors = ['green'] * len(acc_df)

        # Highlight Paris FC if present
        if 'Paris FC' in acc_df['Team'].values:
            paris_idx = acc_df.index[acc_df['Team'] == 'Paris FC'][0]
            colors[paris_idx] = 'orange'  # couleur spécifique pour Paris FC

        plt.barh(acc_df['Team'], acc_df['Accuracy'], color=colors)
        plt.xlabel('Accuracy')
        plt.ylabel('Team')
        plt.title('Prediction Accuracy by Team')
        plt.gca().invert_yaxis()  # Best team on top
        plt.grid(axis='x', linestyle='--', alpha=0.6)
        plt.tight_layout()
        plt.show()

        return acc_df
