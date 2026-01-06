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
    

    def get_data(self):

        """
        Returns a dictionnary with the list of teams and the result matrix

        Returns
        -------

        """

        # All unique teams
        teams = sorted(pd.unique(self.df[['HomeTeam','AwayTeam']].values.ravel()))
        team_index = {team:i for i, team in enumerate(teams)}

        n = len(teams)

        # Victory Matrix
        W = np.zeros((n,n), dtype=int)

        # Draw Matrix
        D = np.zeros((n,n), dtype=int)

        for _,row in self.df.iterrows():

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