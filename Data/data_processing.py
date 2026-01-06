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
    
