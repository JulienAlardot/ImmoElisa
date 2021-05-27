import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures

pd.options.display.max_colwidth = 300
pd.options.display.max_columns = 30

sns.set("notebook", style="darkgrid")
# df: pd.DataFrame = pd.read_csv("https://raw.githubusercontent.com/JulienAlardot/ImmoElizaVisu/main/clean_database.csv", index_col=0)
# df.to_csv("database.csv")

df = pd.read_csv("database.csv")

X = df.drop(columns=["Price"]).values
y = df["Price"].values
X_train, X_test, y_train, y_test = train_test_split(X, y)
degree = 1
scores = list()
for pca_component in range(1, min(df.shape)):
    steps = (
        # ("poly_fear", PolynomialFeatures(degree=degree, interaction_only=False)),
        ("pca", PCA(pca_component)),
        ("clf_model", LinearRegression(n_jobs=os.cpu_count() * 3 // 4)),)

    pipe = Pipeline(steps)
    pipe.fit(X_train, y_train)
    scores.append((pca_component, pipe.score(X_train, y_train), pipe.score(X_test, y_test)))

scores_df = pd.DataFrame(scores, columns=["number_of_pca_components", "training_score", "test_score"])
sns.lineplot(data=scores_df, x="number_of_pca_components")