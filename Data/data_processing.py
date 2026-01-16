#========== Module for Data Processing ==========#

## Imports
import numpy as np
import pandas as pd

#================================================#


class DataProcesser:

    """
    Class for processing data into a correct input for models
    """

    def __init__(self, filepath : str):

        """
        Data Processer

        Parameters
        ----------
        filepath : str
            Path to the data file
        """

        self.df = pd.read_csv(filepath)

        self.train_df : pd.DataFrame = None
        self.test_df : pd.DataFrame = None

        return
    
    
    def split_train_test(self, test_season : str):
        
        """
        Split the data into training and test sets based on seasons
    
        Parameters
        ----------
        test_season : str
            Season to use for testing (format 'YY-YY'). All seasons before this one will be used for training.
            Seasons after this one will be ignored.
                    
        Returns
        -------
        train_df : pd.DataFrame
            Training data (all seasons before test_season)
        test_df : pd.DataFrame
            Test data (matches from test_season)
        """

        # Extract the starting year of the test season
        target_year = int(test_season.split('-')[0])

        # Create train and test dataframes
        train_df = self.df[self.df['Season'].apply(lambda s: int(s.split('-')[0]) < target_year)].copy()
        test_df = self.df[self.df['Season'] == test_season].copy()

        # Store the splits
        self.train_df = train_df
        self.test_df = test_df

        return train_df, test_df
    

    def get_data_BT(self, df : pd.DataFrame = None):

        """
        Returns a dictionnary with the list of teams and the result matrix

        Parameters
        ----------
        df : pd.DataFrame. Default = None
            Dataframe to process. If None, uses the main dataframe.

        Returns
        -------
        data : dict
            Dictionary with keys:
                'Teams' : list of team names
                'Victory Matrix' : np.array of shape (n_teams, n_teams) with number of victories
                'Draw Matrix' : np.array of shape (n_teams, n_teams) with number of draws

        """

        # Provided dataframe or main dataframe
        data_df = df if df is not None else self.df

        # All unique teams
        teams = sorted(pd.unique(data_df[['HomeTeam','AwayTeam']].values.ravel()))
        team_index = {team:i for i, team in enumerate(teams)}

        n = len(teams)

        # Victory Matrix
        W = np.zeros((n,n), dtype=int)

        # Draw Matrix
        D = np.zeros((n,n), dtype=int)

        for _,row in data_df.iterrows():

            i = team_index[row['HomeTeam']]
            j = team_index[row['AwayTeam']]

            if row['FTR'] == 'H':
                W[i,j] += 1
            elif row['FTR'] == 'A':
                W[j,i] += 1
            elif row['FTR'] == 'D':
                D[i,j] += 1
                D[j,i] += 1


        data = {'Teams': teams,
                'Victory Matrix': W,
                'Draw Matrix': D}

        return data
    
    
    def get_train_test_data_BT(self):
        
        """
        Returns processed training and test data
        
        Returns
        -------
        train_data : dict
            Processed training data with Teams, Victory Matrix, Draw Matrix
            {'Teams': list of team names,
             'Victory Matrix': np.array of shape (n_teams, n_teams) with number of victories,
             'Draw Matrix': np.array of shape (n_teams, n_teams) with number of draws}
        test_data : dict
            Processed test data with Teams, Victory Matrix, Draw Matrix
            {'Teams': list of team names,
             'Victory Matrix': np.array of shape (n_teams, n_teams) with number of victories,
             'Draw Matrix': np.array of shape (n_teams, n_teams) with number of draws}
        """

        if (self.train_df is not None) and (self.test_df is not None):
            
            # Process both splits
            train_data = self.get_data_BT(self.train_df)
            test_data = self.get_data_BT(self.test_df)
                
            return train_data, test_data
        
        else:
            raise ValueError("Train and Test dataframes are not set. Please run split_train_test() first.")
    

    def get_data_NBTR(self, feature_names : list) -> dict:

        """
        Returns processed training data for NBTR model

        Parameters
        ----------
        feature_names : list
            List of feature column names (without _Home or _Away suffix)
        
        Returns
        -------
        data : dict
            Dictionary with keys:
                'X_home' : np.array of shape (n_samples, n_features)
                'X_away' : np.array of shape (n_samples, n_features)
                'y' : np.array of shape (n_samples,)
                'input_dim' : int, number of features
                'seasons' : np.array of shape (n_samples,), seasons for each match
        """

        # Seasons used in training
        seasons = self.train_df['Season'].values

        # Features and target 
        X_home = self.train_df[[f"{col}_Home" for col in feature_names]].values.astype(np.float32)
        X_away = self.train_df[[f"{col}_Away" for col in feature_names]].values.astype(np.float32)
        y = self.train_df['Target'].values.astype(np.float32)

        # Input dimension
        input_dim = len(feature_names)

        return {'X_home' : X_home,
                'X_away' : X_away,
                'y' : y,
                'input_dim' : input_dim,
                'seasons' : seasons}
