#========== Data Preparing Module ==========#

## Imports
import numpy as np
import pandas as pd
from scipy import stats

#===========================================#

class DataPreparer:

    """
    Docstring for DataPreparer
    """

    def __init__(self, match_data_filepath : str, market_value_filepath : str):

        """
        Docstring for __init__
        
        :param self: Description
        :param filepath: Description
        :type filepath: str
        """

        # Original Match Data        
        self.df = pd.read_csv(match_data_filepath)

        # Market Value Data
        self.market_value_df = pd.read_csv(market_value_filepath)

        # New Features from Feature Engineering
        self.features_df : pd.DataFrame = None

        # Team Mapping between datasets
        self.team_mapping = {
            "Ajaccio GFCO": "GFC Ajaccio", "Toulouse FC": "FC Toulouse", "Toulouse": "FC Toulouse",   
            "Arles": "AC Arles-Avignon", "Arles-Avignon": "AC Arles-Avignon", 
            "Evian Thonon Gaillard": "FC Évian Thonon Gaillard", "Evian TG": "Evian Thonon Gaillard FC",
            "Girondins Bordeaux": "FC Girondins Bordeaux", "Bordeaux": "FC Girondins Bordeaux",
            "Paris SG": "Paris Saint-Germain", "PSG": "Paris Saint-Germain",
            "St Etienne": "AS Saint-Étienne", "Saint-Étienne": "AS Saint-Étienne",
            "Lille": "LOSC Lille", "Lyon": "Olympique Lyon", "Marseille": "Olympique Marseille", 
            "Monaco": "AS Monaco", "Nice": "OGC Nice", "Rennes": "Stade Rennais FC", "Nantes": "FC Nantes",
            "Reims": "Stade Reims", "Montpellier": "Montpellier HSC", "Lorient": "FC Lorient",
            "Lens": "RC Lens", "Brest": "Stade Brestois 29", "Strasbourg": "RC Strasbourg Alsace", 
            "Metz": "FC Metz", "Troyes": "ESTAC Troyes", "Dijon": "Dijon FCO", "Angers": "Angers SCO", 
            "Ajaccio": "AC Ajaccio", "Gazelec Ajaccio": "GFC Ajaccio", "Bastia": "SC Bastia",
            "Nancy": "AS Nancy-Lorraine", "Caen": "SM Caen", "Guingamp": "EA Guingamp", 
            "Auxerre": "AJ Auxerre", "Sochaux": "FC Sochaux-Montbéliard", "Le Mans": "Le Mans FC",
            "Valenciennes": "Valenciennes FC", "Clermont": "Clermont Foot 63", "Amiens": "Amiens SC", 
            "Nimes": "Nîmes Olympique", "Le Havre": "Le Havre AC", "Boulogne": "US Boulogne"
        }

        return
    
    
    def feature_engineering(self, save : bool = False, display : bool = True):

        """
        Perform feature engineering on the dataset

        Parameters
        ----------
        save : bool. Default = False
            Whether to save the resulting features dataframe as a CSV file
        display : bool. Default = True
            Whether to return the resulting features dataframe when calling method
            
        Returns
        -------
        df_features : pd.DataFrame
            Dataframe containing the engineered features
        """

        df = self.df.copy()

        df['HomePts'] = df.apply(lambda x: 3 if x['FTR'] == 'H' else (1 if x['FTR'] == 'D' else 0), axis=1)
        df['AwayPts'] = df.apply(lambda x: 3 if x['FTR'] == 'A' else (1 if x['FTR'] == 'D' else 0), axis=1)

        df_home = df[['Season', 'HomeTeam', 'HomePts', 'FTHG', 'FTAG']].rename(
            columns={'HomeTeam': 'Team', 'HomePts': 'Points', 'FTHG': 'Goals_Scored', 'FTAG': 'Goals_Conceded'}
        )
        df_home['IsHome'] = 1

        df_away = df[['Season', 'AwayTeam', 'AwayPts', 'FTAG', 'FTHG']].rename(
            columns={'AwayTeam': 'Team', 'AwayPts': 'Points', 'FTAG': 'Goals_Scored', 'FTHG': 'Goals_Conceded'}
        )
        df_away['IsHome'] = 0

        df_all = pd.concat([df_home, df_away], ignore_index=True)

        df_stats = df_all.groupby(['Season', 'Team'], as_index=False).agg({
            'Points': 'sum',
            'Goals_Scored': 'sum',
            'Goals_Conceded': 'sum',
            'IsHome': 'count'
        })
        df_stats = df_stats.rename(columns={'IsHome': 'Games_Played'})

        home_points = df_all[df_all['IsHome'] == 1].groupby(['Season', 'Team'])['Points'].sum().reset_index()
        home_points = home_points.rename(columns={'Points': 'Home_Points_Total'})

        df_stats = pd.merge(df_stats, home_points, on=['Season', 'Team'], how='left').fillna(0)

        # Feature engineering part

        # Classical means
        df_stats['Avg_Goals_Scored'] = df_stats['Goals_Scored'] / df_stats['Games_Played']
        df_stats['Avg_Goals_Conceded'] = df_stats['Goals_Conceded'] / df_stats['Games_Played']
        df_stats['Goal_Difference'] = df_stats['Goals_Scored'] - df_stats['Goals_Conceded']
        
        #  Pythagorean Expectation (Formule : G^2 / (G^2 + GA^2))
        df_stats['Pythagorean_Exp'] = (df_stats['Goals_Scored']**2) / (
            (df_stats['Goals_Scored']**2) + (df_stats['Goals_Conceded']**2)
        )

        # Home dependency
        df_stats['Home_Dependency'] = df_stats['Home_Points_Total'] / df_stats['Points']
        df_stats['Home_Dependency'] = df_stats['Home_Dependency'].fillna(0.5)

        # Ranks in the previous seasons
        df_stats = df_stats.sort_values(by=['Season', 'Points'], ascending=[True, False])
        df_stats['Rank'] = df_stats.groupby('Season')['Points'].rank(method='first', ascending=False).astype(int)
        df_stats['Strength_Rank_Based'] = 1 - (df_stats['Rank'] / 21)

        final_cols = [
            'Season', 'Team',
            'Rank', 'Points',
            'Strength_Rank_Based',
            'Pythagorean_Exp',
            'Avg_Goals_Scored',
            'Avg_Goals_Conceded',
            'Home_Dependency'
        ]

        if df_stats is not None:
            self.features_df = df_stats[final_cols]
        else:
            raise ValueError("Error while creating the final features dataframe")

        if save:
            self.features_df.to_csv("Data/Datasets/features_df.csv", index=False)

        if display:
            return self.features_df
        return
    

    def _match_prep(self):

        """
        Docstring
        """

        # Add a 'Previous Season' Column
        def _get_prev_season(season : str):
            """Compute the previous season string in YY-YY format"""
            try:
                start = int(season.split('-')[0])
                prev_start = (start - 1) % 100
                prev_end = start
                return f"{prev_start:02d}-{prev_end:02d}"
            except:
                return None

        if 'Season' in self.df.columns:
            self.df['Prev_Season'] = self.df['Season'].apply(_get_prev_season)
        else:
            raise IndexError("'Season' not in match_df.columns")

        return


    def _market_value_prep(self):

        """
        Docstring
        """

        # Standard Season name
        def _format_season(s:str):
            """Convert YYYY-YYYY format to YY-YY format"""
            s = s.strip()
            parts = s.split('-')
            return '-'.join([p[-2:] for p in parts])
        
        if 'Season' in self.market_value_df.columns:
            self.market_value_df['Season'] = self.market_value_df['Season'].astype(str).apply(_format_season)
        else:
            raise ValueError("'Season' not in market_value_df.columns")
        
        #
        if 'Total_Market_Value_Millions' in self.market_value_df.columns:
            self.market_value_df = self.market_value_df.rename(columns={'Total_Market_Value_Millions': 'Market_Value'})
        elif 'Value' in self.market_value_df.columns:
            self.market_value_df = self.market_value_df.rename(columns={'Value': 'Market_Value'})

        return


    def _team_mapping(self):

        """
        Docstring
        """
        
        # Creating feature engineering if not done
        if self.features_df is None:
            self.feature_engineering(save=False, display=False)

        # Mapping team names

        for df in [self.df, self.features_df, self.market_value_df]:
            if 'HomeTeam' in df.columns: df['HomeTeam'] = df['HomeTeam'].replace(self.team_mapping)
            if 'AwayTeam' in df.columns: df['AwayTeam'] = df['AwayTeam'].replace(self.team_mapping)
            if   'Team'   in df.columns:   df['Team']   = df['Team'].replace(self.team_mapping)

        return
    

    def merge_df(self, save : bool = False, display : bool = True):

        """
        Merge match data with market value and engineered features

        Parameters
        ----------
        save : bool. Default = False
            Whether to save the resulting merged dataframe as a CSV file
        display : bool. Default = True
            Whether to return the resulting merged dataframe when calling method
        
        Returns
        -------
        df_merged : pd.DataFrame
            Merged dataframe with enriched features
        """

        # Preparing Datasets
        self._match_prep()
        self._market_value_prep()
        self._team_mapping()

        # Adding Market value to Home teams
        self.df_merged = pd.merge(self.df, self.market_value_df[['Season', 'Team', 'Market_Value']],
                                  left_on=['Season', 'HomeTeam'], right_on=['Season', 'Team'], how='left')
        self.df_merged = self.df_merged.rename(columns={'Market_Value': 'Market_Value_Home'}).drop(columns=['Team'])

        # Adding Market value to Away teams
        self.df_merged = pd.merge(self.df_merged, self.market_value_df[['Season', 'Team', 'Market_Value']],
                                  left_on=['Season', 'AwayTeam'], right_on=['Season', 'Team'], how='left')
        self.df_merged = self.df_merged.rename(columns={'Market_Value': 'Market_Value_Away'}).drop(columns=['Team'])

        # Fill Market Value NaNs
        self.df_merged['Market_Value_Home'] = self.df_merged['Market_Value_Home'].fillna(10)
        self.df_merged['Market_Value_Away'] = self.df_merged['Market_Value_Away'].fillna(10)

        # Historical Stats Enrichment
        cols_features = ['Season', 'Team', 'Strength_Rank_Based', 'Pythagorean_Exp', 'Avg_Goals_Scored', 'Home_Dependency']
        feature_names = ['Strength_Rank_Based', 'Pythagorean_Exp', 'Avg_Goals_Scored', 'Home_Dependency']

        # Merge Home Stats
        self.df_merged = pd.merge(self.df_merged, self.features_df[cols_features],
                                  left_on=['Prev_Season', 'HomeTeam'], right_on=['Season', 'Team'], how='left')
        
        if 'Season_x' in self.df_merged.columns: self.df_merged = self.df_merged.rename(columns={'Season_x': 'Season'})
        if 'Season_y' in self.df_merged.columns: self.df_merged = self.df_merged.drop(columns=['Season_y'])
        if 'Team' in self.df_merged.columns: self.df_merged = self.df_merged.drop(columns=['Team'])

        rename_dict = {col: col + '_Home' for col in feature_names}
        self.df_merged = self.df_merged.rename(columns=rename_dict)

        self.df_merged = pd.merge(self.df_merged, self.features_df[cols_features],
                                  left_on=['Prev_Season', 'AwayTeam'], right_on=['Season', 'Team'], how='left')
        
        # Cleaning
        if 'Season_x' in self.df_merged.columns: self.df_merged = self.df_merged.rename(columns={'Season_x': 'Season'})
        if 'Season_y' in self.df_merged.columns: self.df_merged = self.df_merged.drop(columns=['Season_y'])
        if 'Team' in self.df_merged.columns: self.df_merged = self.df_merged.drop(columns=['Team'])

        rename_dict = {col: col + '_Away' for col in feature_names}
        self.df_merged = self.df_merged.rename(columns=rename_dict)

        # Fill Stats NaNs (Promoted teams)
        defaults = {'Strength_Rank_Based': 0.05, 'Pythagorean_Exp': 0.35, 'Avg_Goals_Scored': 0.8, 'Home_Dependency': 0.5}
        for col in feature_names:
            self.df_merged[col + '_Home'] = self.df_merged[col + '_Home'].fillna(defaults[col])
            self.df_merged[col + '_Away'] = self.df_merged[col + '_Away'].fillna(defaults[col])
        
        # Target Variables
        self.df_merged['Target'] = self.df_merged['FTR'].map({'H': 1.0, 'A': 0.0, 'D': 0.5})
        self.df_merged = self.df_merged.dropna(subset=['Target'])
        self.df_merged['Log_Market_Value_Home'] = np.log1p(self.df_merged['Market_Value_Home'])
        self.df_merged['Log_Market_Value_Away'] = np.log1p(self.df_merged['Market_Value_Away'])

        # Line 150
        def _get_season_stats(df_matchs : pd.DataFrame):
            """Calculate agregated stats (Shots, Corners, Cards) per team per season."""

            home = df_matchs.groupby(['Season', 'HomeTeam']).agg({
                'HST': 'sum', 'HC': 'sum', 'HF': 'sum', 'HY': 'sum', 'HR': 'sum', 'FTR': 'count'
            }).reset_index() 
            home = home.rename(columns={'HomeTeam': 'Team', 'FTR': 'Games'})

            away = df_matchs.groupby(['Season', 'AwayTeam']).agg({
                'AST': 'sum', 'AC': 'sum', 'AF': 'sum', 'AY': 'sum', 'AR': 'sum', 'FTR': 'count'
            }).reset_index()
            away = away.rename(columns={'AwayTeam': 'Team', 'FTR': 'Games', 
                                'AST': 'HST', 'AC': 'HC', 'AF': 'HF', 'AY': 'HY', 'AR': 'HR'}) 

            total = pd.concat([home, away])
            stats = total.groupby(['Season', 'Team']).sum().reset_index()

            stats['Avg_Shots_Target'] = stats['HST'] / stats['Games']
            stats['Avg_Corners'] = stats['HC'] / stats['Games']
            stats['Avg_Cards'] = (stats['HY'] + 3*stats['HR']) / stats['Games']
    
            return stats[['Season', 'Team', 'Avg_Shots_Target', 'Avg_Corners', 'Avg_Cards']]
        
        df_game_stats = _get_season_stats(self.df_merged)
        self.df_merged = pd.merge(self.df_merged, df_game_stats, left_on=['Prev_Season', 'HomeTeam'], right_on=['Season', 'Team'], how='left')

        # New cleaning
        if 'Season_x' in self.df_merged.columns: self.df_merged = self.df_merged.rename(columns={'Season_x': 'Season'})
        if 'Season_y' in self.df_merged.columns: self.df_merged = self.df_merged.drop(columns=['Season_y'])
        if 'Team' in self.df_merged.columns: self.df_merged = self.df_merged.drop(columns=['Team'])

        self.df_merged = self.df_merged.rename(columns={
            'Avg_Shots_Target': 'Avg_Shots_Target_Home',
            'Avg_Corners': 'Avg_Corners_Home',
            'Avg_Cards': 'Avg_Cards_Home'
        })

        # Merge Away Game Stats
        self.df_merged = pd.merge(self.df_merged, df_game_stats, left_on=['Prev_Season', 'AwayTeam'], right_on=['Season', 'Team'], how='left')

        # New cleaning
        if 'Season_x' in self.df_merged.columns: self.df_merged = self.df_merged.rename(columns={'Season_x': 'Season'})
        if 'Season_y' in self.df_merged.columns: self.df_merged = self.df_merged.drop(columns=['Season_y'])
        if 'Team' in self.df_merged.columns: self.df_merged = self.df_merged.drop(columns=['Team'])

        self.df_merged = self.df_merged.rename(columns={
            'Avg_Shots_Target': 'Avg_Shots_Target_Away',
            'Avg_Corners': 'Avg_Corners_Away',
            'Avg_Cards': 'Avg_Cards_Away'
        })

        # Fill NaNs for game stats
        cols_new = ['Avg_Shots_Target', 'Avg_Corners', 'Avg_Cards']
        for c in cols_new:
            if f"{c}_Home" in self.df_merged.columns:
                mean_val = self.df_merged[f"{c}_Home"].mean()
                self.df_merged[f"{c}_Home"] = self.df_merged[f"{c}_Home"].fillna(mean_val)
                self.df_merged[f"{c}_Away"] = self.df_merged[f"{c}_Away"].fillna(mean_val)

        # Output
        if save:
            print("Processing complete. Saving merged dataframe to 'Complete_df.csv'.")
            self.df_merged.to_csv("Data/Datasets/Complete_df.csv", index=False)
        else:
            print("Processing complete. Merged dataframe is ready.")
        
        if display:
            return self.df_merged
        return

            