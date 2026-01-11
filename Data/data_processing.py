#========== Data Processing ==========#

## Imports
import numpy as np
import pandas as pd

#=====================================#

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

    
    def filter_year(self, year : str):

        """
        Filter data for a given year

        Parameters
        ----------
        year : str
            Year to filter the data on (format: "YY-YY" or "YYYY-YYYY")
        """

        self.df = self.df[self.df['Season']==year]

        return self
    
    
    def split_train_test(self, test_size : int, nb_teams : int = 20):
        
        """
        Split the data into training and test sets
    
        Parameters
        ----------
        test_size : int
            Number of match days to include in the test set
        
        nb_teams : int. Default = 20
            Number of teams in the league (used to calculate number of matches per day)
            
        Returns
        -------
        train_df : pd.DataFrame
            Training data
        test_df : pd.DataFrame
            Test data
        """

        # Create a copy of the dataframe
        df_copy = self.df.copy()
                
        # Calculate split index
        split_idx = int(len(df_copy) - test_size * nb_teams /2)
        
        # Split the data
        train_df = df_copy.iloc[:split_idx].copy()
        test_df = df_copy.iloc[split_idx:].copy()

        # Store the splits
        self.train_df = train_df
        self.test_df = test_df

        return train_df, test_df
    

    def get_data(self, df : pd.DataFrame = None):

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
    
    
    def get_train_test_data(self):
        
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
            train_data = self.get_data(self.train_df)
            test_data = self.get_data(self.test_df)
                
            return train_data, test_data
        
        else:
            raise ValueError("Train and Test dataframes are not set. Please run split_train_test() first.")
    
    
