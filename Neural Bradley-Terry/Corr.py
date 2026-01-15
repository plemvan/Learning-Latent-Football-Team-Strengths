import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("Data/dataset_training.csv")
feature_names = [
    'Log_Market_Value',     
    'Strength_Rank_Based',   
    'Pythagorean_Exp',       
    'Avg_Goals_Scored',      
    'Home_Dependency',
    'Avg_Shots_Target',  
    'Avg_Corners',       
    'Avg_Cards'        
]

cols_to_analyze = [f"{col}_Home" for col in feature_names]
corr_matrix = df[cols_to_analyze].corr()
corr_matrix.columns = feature_names
corr_matrix.index = feature_names


plt.figure(figsize=(12, 10))
sns.heatmap(
    corr_matrix, 
    annot=True,        
    fmt=".2f",         
    cmap='coolwarm',    
    vmin=-1, vmax=1,    
    linewidths=0.5,
    cbar_kws={"shrink": .8},
    square=True)
plt.title("Model Features Correlation Matrix", fontsize=16, pad=20)
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()

