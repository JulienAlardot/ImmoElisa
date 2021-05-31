import os
from typing import List, Tuple, Dict, Union
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline

pd.options.display.max_colwidth = 300
pd.options.display.max_columns = 30

sns.set("notebook", style="darkgrid")
# df: pd.DataFrame = pd.read_csv("https://raw.githubusercontent.com/JulienAlardot/ImmoElizaVisu/main/clean_database.csv", index_col=0)
# df.to_csv("database.csv")

# fig = plt.figure(figsize=(10, (max_degree - min_degree) * 10), dpi=72)
# max_pca = min(df.shape) - 1
# pca_progress = (degree - 1) / max_degree + (pca_component / max_pca) / max_degree


df: pd.DataFrame = pd.read_csv("database.csv")
X: np.ndarray = df.drop(columns=["Price"]).values
y: np.ndarray = df["Price"].values
X_train, X_test, y_train, y_test = train_test_split(X, y)

steps: Tuple[Tuple] = (
    ("reg", GradientBoostingRegressor()),)

pipe: Pipeline = Pipeline(steps=steps)
# {'reg__learning_rate': 0.1, 'reg__max_depth': 5, 'reg__min_samples_leaf': 10, 'reg__min_samples_split': 10, 'reg__n_estimators': 400, 'reg__n_iter_no_change': 100, 'reg__random_state': 42}
param_grid: Dict[str, Union[List, np.ndarray]] = {
    "reg__n_estimators": np.linspace(300, 500, 4).astype(np.int32)[::-1],
    'reg__learning_rate': np.linspace(0.05, 0.2, 4)[::-1],
    'reg__max_depth': np.linspace(1, 6, 4).astype(np.int32)[::-1],
    'reg__n_iter_no_change': np.linspace(50, 150, 4).astype(np.int32)[::-1],
    'reg__min_samples_split': (10 ** np.linspace(1, 2, 2)).astype(np.int32),
    'reg__min_samples_leaf': np.linspace(1, 50, 4).astype(np.int32),
    "reg__random_state": [42]
}

gs: GridSearchCV = GridSearchCV(estimator=pipe, param_grid=param_grid, n_jobs=os.cpu_count() * 7 // 8, cv=3, verbose=3)
gs.fit(X_train, y_train)



print(gs.best_params_)



# scores.append(pipe.score(X_train, y_train), pipe.score(X_test, y_test)))
# scores_df = pd.DataFrame(scores, columns=["degree", "number_of_pca_components", "training_score", "test_score"])
# ax = fig.add_subplot(max_degree-min_degree+1, 1, degree-min_degree+1)
# sns.lineplot(data=scores_df, x="number_of_pca_components", y="training_score", color="red", label="Training Score",
#      ax=ax)
# sns.lineplot(data=scores_df, x="number_of_pca_components", y="test_score", color="darkblue", label="Test Score",
#      ax=ax)
# plt.title(f"{degree} Polynomial Features")
# plt.ylabel("Score ( max= 1.0 )")
# plt.xlabel("PCA Components")
# plt.ylim(0, 1)
# plt.legend()
# print("")
# plt.suptitle("Hyper Parameter Search for Immo Eliza Database")
# plt.tight_layout()
# plt.show()
