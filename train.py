import os
import time
from typing import List

import numpy as np
import pandas as pd
import seaborn as sns
import xgboost as xgb
from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.pipeline import Pipeline

pd.options.display.max_colwidth = 300
pd.options.display.max_columns = 30
sns.set("notebook", style="darkgrid")


def clean_df(df):
    """
    Function used to convert the database in a version +-30% the weight of the original
    :param df: the original version of the database Dataframe
    :type df: pd.DataFrame
    :return: a lightweight version of the database DataFrame (without dataloss)
    :rtype: pd.DataFrame
    """
    property_subtype = (
        "penthouse",
        "building",
        "studio",
        "duplex",
        "triplex",
        "loft",
        "ground floor",
        "student",
        "investment property",
        "villa",
        "mansion",
        "mixed",
        "apartments row",
        "farmhouse",
        "cottage",
        "floor",
        "town",
        "service flat",
        "manor",
        "castle",
        "pavilion",
    )
    df["Source"] = pd.Categorical(df["Source"])
    # for column in df.columns:
    #     print(column, df[column].unique())

    df["Type of property"] = pd.Categorical(
        df["Type of property"], categories=("apartment", "house"), ordered=True
    )
    df["Type of sale"] = pd.Categorical(
        df["Type of sale"], categories=("regular sale", "public sale")
    )
    df["Subtype of property"] = pd.Categorical(
        df["Subtype of property"], categories=property_subtype
    )
    df["State of the building"] = pd.Categorical(
        df["State of the building"],
        categories=("to renovate", "good", "new"),
        ordered=True,
    )
    df["Province"] = pd.Categorical(
        df["Province"],
        categories=("West-Vlanderen",
                    "Oost-Vlanderen",
                    "Vlaams-Brabant",
                    "Brussels",
                    "Liège",
                    "Hainaut",
                    "Luxembourg",
                    "Namur",
                    "Brabant Wallon",
                    "Limburg",
                    "Antwerp",
                    )
    )
    df["Region"] = pd.Categorical(
        df["Region"],
        categories=("Brussels Capital",
                    "Vlaams",
                    "Wallonie"
                    )
    )

    df["Fully equipped kitchen"] = df["Fully equipped kitchen"].astype(np.float16)
    df["Furnished"] = df["Furnished"].astype(np.float16)
    df["Locality"] = df["Locality"].astype(np.int16)
    df["Open fire"] = df["Open fire"].astype(np.float16)
    df["Swimming pool"] = df["Swimming pool"].astype(np.float16)
    df["Garden"] = df["Garden"].astype(np.float16)
    df["Terrace Area"] = df["Terrace Area"].astype(np.float32)
    df["Surface of the land"] = df["Surface of the land"].astype(np.float32)
    df["Surface area of the plot of land"] = df[
        "Surface area of the plot of land"
    ].astype(np.float32)
    df["Garden Area"] = df["Garden Area"].astype(np.float32)
    df["Number of facades"] = df["Number of facades"].astype(np.float16)
    df["Area"] = df["Area"].astype(np.float32)
    df["Terrace"] = df["Terrace"].astype(np.float32)
    df["Number of rooms"] = df["Number of rooms"].astype(np.float16)
    df["Fully equipped kitchen"] = df["Fully equipped kitchen"].astype(np.float16)
    df.loc[df.Area == df.Area.max(), "Area"] = 172
    df.drop_duplicates(inplace=True)
    df = df.loc[np.logical_or(np.logical_and(1000 <= df["Locality"], df["Locality"] < 10000), df["Locality"].isna())]
    df = df.loc[df["Url"].str.lower().str.find("havre") == -1, :]
    df = df.loc[df["Url"].str.lower().str.find("paris") == -1, :]
    df = df.loc[df["Url"].str.lower().str.find("lille") == -1, :]
    df = df.loc[np.logical_or(df["Price"] >= 10 ** 4, df["Price"].isna()), :]
    df.loc[df["Garden"] == 0, "Garden Area"] = 0
    df.loc[df["Terrace"] == 0, "Terrace Area"] = 0
    df.where(df["Number of rooms"] >= 1, inplace=True)
    df = df.loc[np.logical_or(df["Number of rooms"] < 125, df["Number of rooms"].isna()),
         :]  # Offers not available anymore or wrongly encoded
    df = df.loc[np.logical_or(df["Area"] >= 11, df["Area"].isna()), :]
    df: pd.DataFrame = df.loc[np.logical_or(df["Area"] < 5000, df["Area"].isna()), :]
    df.where(df["Number of facades"] > 0, inplace=True)
    df.dropna(how="all", inplace=True)
    bad_columns = df.isna().sum()[(df.isna().sum() / df.shape[0] * 100) > 75].sort_values(ascending=False).index
    df.drop(columns=bad_columns, inplace=True)
    df.drop(columns=['Garden', 'Terrace', 'Type of sale', 'Url'], inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


df: pd.DataFrame = pd.read_csv(
    'https://raw.githubusercontent.com/JulienAlardot/challenge-collecting-data/main/Data/database.csv', index_col=0,
    low_memory=False)

df = clean_df(df)
df.to_csv("database.csv")

df: pd.DataFrame = pd.read_csv("database.csv", index_col=0)
df: pd.DataFrame = pd.read_csv("https://raw.githubusercontent.com/JulienAlardot/ImmoElizaVisu/main/clean_database.csv",
                               index_col=0)
# df["Locality"] = df["Locality"].astype(str)
# df = pd.get_dummies(df, drop_first=True)
columns = df.columns
# print(columns)
df: np.ndarray = KNNImputer(n_neighbors=5).fit_transform(df.values)
np.save("database.npy", df)
df = pd.DataFrame(np.load("database.npy"), columns=columns)
X: np.ndarray = df.drop(columns=["Price"]).values
y: np.ndarray = df["Price"].values
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

steps: List = [
    # ("reg", GradientBoostingRegressor()),)
    ("reg", xgb.XGBRegressor(tree_method="gpu_hist", gpu_id=0)), ]

pipe: Pipeline = Pipeline(steps=steps)

# param_grid: Dict[str, Union[List, np.ndarray]] = {
#     "reg__n_estimators": np.linspace(400, 500, 4).astype(np.int32)[::-1],
#     'reg__learning_rate': np.linspace(0.05, 0.1, 3)[::-1],
#     'reg__max_depth': np.linspace(4, 7, 4).astype(np.int32)[::-1],
#     'reg__n_iter_no_change': np.linspace(100, 300, 3).astype(np.int32)[::-1],
#     'reg__min_samples_split': (10 ** np.linspace(0.7, 1.4, 2)).astype(np.int32),
#     'reg__min_samples_leaf': np.linspace(5, 20, 3).astype(np.int32),
#     "reg__random_state": [42]
# }
n_iter = (30, 1)
# 0.8785173377264895
# 0.7157823231265649
# {'reg__subsample': 0.9797222005293017, 'reg__random_state': 42, 'reg__n_estimators': 328, 'reg__max_depth': 5,
#  'reg__learning_rate': 0.19463485960181195, 'reg__gamma': 0.14790626224466663,
#  'reg__colsample_bytree': 0.8702556952587032, 'reg__colsample_bylevel': 0.4990910003749124}

# 0.87868669132923
# 0.7210963293130916
# {'reg__subsample': 0.9132431570611518, 'reg__random_state': 42, 'reg__n_estimators': 368, 'reg__max_depth': 5,
#  'reg__learning_rate': 0.13752665971992017, 'reg__gamma': 0.2792546800806579,
#  'reg__colsample_bytree': 0.821384321745897, 'reg__colsample_bylevel': 0.9520802990142427}

# 0.8346554607329777
# 0.7680063492934566
# {'reg__random_state': 42, 'reg__n_estimators': 282, 'reg__max_depth': 3, 'reg__learning_rate': 0.3519464421105537,
#  'reg__gamma': 0.43511904972260096, 'reg__colsample_bytree': 0.863385123641318}

# 0.8620983966280047
# 0.7779025085957194
# {'reg__random_state': 42, 'reg__n_estimators': 398, 'reg__max_depth': 4, 'reg__learning_rate': 0.17031374854894699,
#  'reg__gamma': 0.782939727865437, 'reg__colsample_bytree': 0.4935127851580509}
# 0.8686092117675117
# 0.7772824406934563
# {'reg__random_state': 42, 'reg__n_estimators': 398, 'reg__max_depth': 4, 'reg__learning_rate': 0.1936770956518794,
#  'reg__gamma': 0.47425566971097893, 'reg__colsample_bytree': 0.49031824448203354}

# 0.8827566437559508
# 0.7879262473989432
# {'reg__random_state': 42, 'reg__n_estimators': 390, 'reg__max_depth': 4, 'reg__learning_rate': 0.22583393758762554,
#  'reg__gamma': 0.7370113058359546, 'reg__colsample_bytree': 0.670357922704685}

# 0.8728306473999864
# 0.7886444873001949
# {'reg__random_state': 42, 'reg__n_estimators': 372, 'reg__max_depth': 4, 'reg__learning_rate': 0.19244881624366028,
#  'reg__gamma': 0.7983684048959909, 'reg__colsample_bytree': 0.6338551950579786}

# 0.917247680296272
# 0.7926032741911082
# {'reg__random_state': 42, 'reg__n_estimators': 393, 'reg__max_depth': 5, 'reg__learning_rate': 0.18096184030841525,
#  'reg__gamma': 0.6223292520026902, 'reg__colsample_bytree': 0.6672209435032146}
params = {
    # 'reg__colsample_bylevel': np.random.uniform(0.1, 1., n_iter).flatten(),
    'reg__colsample_bytree': np.random.uniform(0.5, 0.7, n_iter).flatten(),
    "reg__gamma": np.random.uniform(.4, 0.8, n_iter).flatten(),
    "reg__learning_rate": np.random.uniform(0.12, 0.2, n_iter).flatten(),  # default 0.1
    "reg__max_depth": np.random.randint(4, 6, 2).flatten(),  # default 3
    "reg__n_estimators": np.random.randint(300, 400, n_iter).flatten(),  # default 100
    # "reg__subsample": np.random.uniform(0.1, 1., n_iter).flatten(),
    "reg__random_state": [42]

}
print(time.ctime())
gs: RandomizedSearchCV = RandomizedSearchCV(estimator=pipe, n_iter=200, param_distributions=params,
                                            n_jobs=os.cpu_count() * 4 // 8, cv=3, verbose=3)
gs.fit(X_train, y_train)
print(time.ctime())
# pipe.fit(X_train, y_train)

print(gs.score(X_train, y_train))
print(gs.score(X_test, y_test))

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
