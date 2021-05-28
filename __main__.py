import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures

pd.options.display.max_colwidth = 300
pd.options.display.max_columns = 30

sns.set("notebook", style="darkgrid")
# df: pd.DataFrame = pd.read_csv("https://raw.githubusercontent.com/JulienAlardot/ImmoElizaVisu/main/clean_database.csv", index_col=0)
# df.to_csv("database.csv")

df = pd.read_csv("database.csv")

X = df.drop(columns=["Price"]).values
y = df["Price"].values
X_train, X_test, y_train, y_test = train_test_split(X, y)
min_degree = 2
max_degree = 4
fig = plt.figure(figsize=(10, (max_degree-min_degree)*10), dpi=72)
min_pca = 1
max_pca = min(df.shape) - 1
for degree in range(min_degree, max_degree + 1):
    scores = list()
    for pca_component in range(min_pca, max_pca + 1, 2):
        pca_progress = (degree - 1) / max_degree + (pca_component / max_pca) / max_degree
        steps = (
            # ("scaler", StandardScaler()),
            ("poly_features", PolynomialFeatures(degree=2, interaction_only=True)),
            # ("pca", PCA(pca_component)),
            ("reg_model", GradientBoostingRegressor(n_estimators=200, learning_rate=pca_progress, max_depth=degree,
                                                    n_iter_no_change=10, min_samples_split=50, min_samples_leaf=50)),)
        pipe = Pipeline(steps)
        print(f"\rProcessing: |{'#' * round(100 * pca_progress)}{'-' * round(100 * (1 - pca_progress))}| " +
              f"Degree search {degree}/{max_degree} | " +
              f"PCA search {pca_component}/{max_pca} | " +
              f"{pca_progress: >5.2%}",
              end="")
        pipe.fit(X_train, y_train)
        scores.append((degree, pca_component, pipe.score(X_train, y_train), pipe.score(X_test, y_test)))
    scores_df = pd.DataFrame(scores, columns=["degree", "number_of_pca_components", "training_score", "test_score"])
    ax = fig.add_subplot(max_degree-min_degree+1, 1, degree-min_degree+1)
    sns.lineplot(data=scores_df, x="number_of_pca_components", y="training_score", color="red", label="Training Score",
                 ax=ax)
    sns.lineplot(data=scores_df, x="number_of_pca_components", y="test_score", color="darkblue", label="Test Score",
                 ax=ax)
    plt.title(f"{degree} Polynomial Features")
    plt.ylabel("Score ( max= 1.0 )")
    plt.xlabel("PCA Components")
    plt.ylim(0, 1)
    plt.legend()
print("")
plt.suptitle("Hyper Parameter Search for Immo Eliza Database")
plt.tight_layout()
plt.show()
