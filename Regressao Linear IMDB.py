import sqlite3
import pandas as pd 
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from math import sqrt

data = pd.read_csv(r"C:\Users\mathe\Downloads\IMDB\imdb_top_5000_tv_shows.csv")

df = data
df_clean = df

print(df_clean.shape)
del df_clean['IMDbLink']
del df_clean['Title_IMDb_Link']
print(df_clean.shape)

df_clean.dropna()
print(F' PRINT PÓS DROPNA: {df_clean.shape}')