

import pandas as pd

from Data.data_processing import Datacleaner

df = pd.read_csv("./Data/ligue1_2010_2025.csv")

interest_columns = ["Div", "Date", "Time", "HomeTeam", "AwayTeam",
                    "FTHG", "HG", "FTAG", "AG", "FTR", "Res",
                    "HTHG", "HTAG", "HTR", "Attendance", "Referee",
                    "HS", "AS", "HST", "AST", "HHW", "AHW",
                    "HC", "AC", "HF", "AF", "HFKC", "AFKC",
                    "HO", "AO", "HY", "AY", "HR", "AR",
                    "HBP", "ABP"]

intersection = [c for c in interest_columns if c in df.columns]
print(intersection)
print(f"Nb colonnes intérêt: {len(interest_columns)}")
print(f"Nb de colonnes d'intérêt dans le DF: {len(intersection)}")

complementaire = [c for c in interest_columns if c not in df.columns]
print(complementaire)

df_clean = df[intersection]
print(df_clean.head())
print(df_clean.size)


cleaner = Datacleaner("./Data/ligue1_2010_2025.csv")

print(cleaner.df)

df_clean_process = cleaner.drop_useless_columns().df
print(df_clean_process)

print(df_clean == df_clean_process)