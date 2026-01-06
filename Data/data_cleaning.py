#========== Data Cleaning and pre-processing ==========#

## Imports
import numpy as np
import pandas as pd
from typing import Union, Literal

#======================================================#


class Datacleaner:
    """Class for the initial cleaning of the database"""

    def __init__(self, filepath: str):

        """
        Data Cleaner

        Parameters
        ----------
        filepath : str
            Path to the data file
        """

        self.df = pd.read_csv(filepath)

        self.interest_columns = ["Div", "Date", "Time", "HomeTeam", "AwayTeam",
                                 "FTHG", "HG", "FTAG", "AG", "FTR", "Res",
                                 "HTHG", "HTAG", "HTR", "Attendance", "Referee",
                                 "HS", "AS", "HST", "AST", "HHW", "AHW",
                                 "HC", "AC", "HF", "AF", "HFKC", "AFKC",
                                 "HO", "AO", "HY", "AY", "HR", "AR",
                                 "HBP", "ABP"]
    

    def drop_useless_columns(self):

        """
        Drops useless columns from the dataset using the list of columns of interest
        """

        self.df = self.df[[col for col in self.interest_columns if col in self.df.columns]]

        return self
    
    
    def drop_rows(self, row_idxs : list[int]):

        """
        Drops rows containing only NaN

        Parameters
        ----------

        row_idxs : List[int]
            List of row indexes
        """

        self.df = self.df.drop(index=row_idxs)

        return self

    

    def fillNA(self, column: str, value : Union[str, int, float]):

        """
        Replace NaN occurences from a column with a given value

        Parameters
        ----------

        column: str
            Name of the column in which the replacement is made
        value : str | int | float
            Value of replacement. Should match the type of the column
        """

        if column not in self.df.columns:
            raise ValueError(f"Column '{column}' does not exist in the dataframe.")

        try:
            self.df[column] = self.df[column].fillna(value)
        except (ValueError, TypeError) as e:
            raise TypeError(f"Impossible to fill '{column}' with {value} : {e}")

        return self


    def add_season(self):
        """
        Add a column that gives the corresponding Ligue1 season
        """

        date_dt = pd.to_datetime(self.df["Date"], format='%d%m%Y', errors="coerce")

        def _season_from_date(date):
            """Inner function to convert date to season"""

            if pd.isna(date):
                return None
            
            year = date.year
            month = date.month

            if month >=7: # July or later -> new season
                return f"{year}-{year+1}"
            else: # June or earlier -> end of previous season
                return f"{year-1}-{year}"
        
        self.df["Season"] = date_dt.apply(_season_from_date)

        return self
    

    def save_data(self, output_path : str, returns : bool = False):

        """
        Save the data set to .csv format
        
        Parameters
        ----------
        output_path : str
            Path for the ouptut .csv file
        returns : bool
            Choice to return or not the DataFrame
        """

        self.df.to_csv(output_path, index=False)

        if returns:
            return self.df
        return


    
