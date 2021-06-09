import json
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
from sklearn.preprocessing import StandardScaler

with open("data/coord.json", "rt", encoding="utf-8") as f:
    geo_data = json.load(f)

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
    # df.where(df["Number of rooms"] >= 1, inplace=True)
    df = df.loc[np.logical_or(df["Number of rooms"] < 125, df["Number of rooms"].isna()),
         :]  # Offers not available anymore or wrongly encoded
    df = df.loc[np.logical_or(df["Area"] >= 11, df["Area"].isna()), :]
    df: pd.DataFrame = df.loc[np.logical_or(df["Area"] < 5000, df["Area"].isna()), :]
    df.where(df["Number of facades"] > 0, inplace=True)
    df.dropna(how="all", inplace=True)
    bad_columns = df.isna().sum()[(df.isna().sum() / df.shape[0] * 100) > 75].sort_values(ascending=False).index
    df.drop(columns=bad_columns, inplace=True)
    df.drop(columns=['Garden', 'Terrace', 'Type of sale', 'Url'], inplace=True)
    print(df.shape)
    df = df.loc[np.isin(df["Locality"].astype(str), geo_data.keys()), :]
    print(df.shape)

    df.reset_index(drop=True, inplace=True)
    return df


df: pd.DataFrame = pd.read_csv(
    'https://raw.githubusercontent.com/JulienAlardot/challenge-collecting-data/main/Data/database.csv', index_col=0,
    low_memory=False)

# df = clean_df(df)
# df.to_csv("database.csv")
# df: pd.DataFrame = pd.read_csv("data/database.csv", index_col=0)

df: pd.DataFrame = pd.read_csv("https://raw.githubusercontent.com/JulienAlardot/ImmoElizaVisu/main/clean_database.csv",
                               index_col=0)

df = df.loc[np.isin(df["Locality"].astype(int).astype(str).values[:], list(geo_data.keys())), :]
df["lat"] = df["Locality"].astype(int).astype(str).apply(lambda x: geo_data[x]["lat"])
df["lng"] = df["Locality"].astype(int).astype(str).apply(lambda x: geo_data[x]["lng"])
df.loc[df["Terrace"] == 0, "Terrace Area"] = 0
df.loc[df["Terrace Area"] < 0, "Terrace Area"] = 0

# df.loc[df["Garden"] == 0, "Garden Area"] = 0
df.drop(columns=['Source_logic-immo.be', "Terrace"], inplace=True)
for column in df.columns:
    if column.lower().startswith("region") or column.lower().startswith("province"):
        df = df.drop(columns=column)

p_c:pd.DataFrame = pd.read_csv("data/postal_codes.csv", sep=";", index_col=0)
p_c.loc[:, "Name"] = p_c.loc[:, "Name"] + ", "
p_c = p_c.groupby("zipcode").sum(numeric_only=False)
for i, row in p_c.iterrows():
    if row[-1].endswith(", "):
        row[-1] = row[-1][:-2]
    p_c.loc[i, :] = row


df_vis = df.copy()
df_vis["Price / m²"] = df["Price"]/df["Area"]

df_count = df_vis.groupby("Locality").count()["Price"].reset_index()
df_count["Count"] = df_count["Price"]

df_count.drop(columns=["Price"], inplace=True)
mean_df = df_vis.groupby(["Locality"]).mean().reset_index()[["Price", "Locality"]]
med_df = df_vis.groupby(["Locality"]).median().reset_index()[["Price", "Locality"]]
prsqrm_mean_df = df_vis.groupby(["Locality"]).mean().reset_index()[["Price / m²", "Locality"]]
prsqrm_median_df = df_vis.groupby(["Locality"]).median().reset_index()[["Price / m²", "Locality"]]
df_vis.groupby("Locality").mean().drop(columns=["Price"]).reset_index(inplace=True)

mean_df["Mean Price"] = np.round(mean_df["Price"],2)
mean_df["Mean Price"] = mean_df["Mean Price"].astype(str) + " €"
mean_df.drop(columns=["Price"], inplace=True)

med_df["Median Price"] = np.round(med_df["Price"], 2)
med_df["Color"] = med_df["Median Price"]
med_df["Median Price/m²"] = prsqrm_median_df["Price / m²"]
med_df["Median Price"] = med_df["Median Price"].astype(str) + " €"
med_df.drop(columns=["Price"], inplace=True)

prsqrm_mean_df["Mean Price / m²"] = np.round(prsqrm_mean_df["Price / m²"], 2)
prsqrm_mean_df["Mean Price / m²"] = prsqrm_mean_df["Mean Price / m²"].astype(str) + " €/m²"
prsqrm_mean_df.drop(columns=["Price / m²"], inplace=True)

prsqrm_median_df["Median Price / m²"] = np.round(prsqrm_median_df["Price / m²"], 2)
prsqrm_median_df["Median Price / m²"] = prsqrm_median_df["Median Price / m²"].astype(str) + " €/m²"
prsqrm_median_df.drop(columns=["Price / m²"], inplace=True)

for dataframe in (df_count, mean_df, med_df, prsqrm_mean_df, prsqrm_median_df):
    print(dataframe)
    df_vis = pd.merge(df_vis, dataframe, "left", "Locality", suffixes=("", ""))

df_vis = pd.merge(df_vis, p_c, "left", left_on="Locality",right_on="zipcode", suffixes=("", ""))

df_vis.to_csv("data/database_visu.csv")
df.drop(columns=['Locality'], inplace=True)

# df["Locality"] = df["Locality"].astype(str)
# df = pd.get_dummies(df, drop_first=True)
columns = df.columns
# print(columns)
df: np.ndarray = KNNImputer(n_neighbors=5).fit_transform(df.values)
np.save("data/database.npy", df)
df = pd.DataFrame(np.load("data/database.npy"), columns=columns)
print(df.shape)

df.to_csv("data/database.csv")
X: np.ndarray = df.drop(columns=["Price"]).values
y: np.ndarray = df["Price"].values
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

steps: List = [
    # ("reg", GradientBoostingRegressor()),)
    ("scaler", StandardScaler()),
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
# 0.9869396159270368
# 0.7801572719679025
# {'reg__random_state': 42, 'reg__n_estimators': 585, 'reg__max_depth': 8, 'reg__learning_rate': 0.17322609079644039,
#  'reg__gamma': 0.9963230807396372, 'reg__colsample_bytree': 0.6909705895347378}
# 0.9052367378216932
# 0.7803710637069743
# {'reg__random_state': 42, 'reg__n_estimators': 404, 'reg__max_depth': 6, 'reg__learning_rate': 0.08422514938256685,
#  'reg__gamma': 0.6292582111150484, 'reg__colsample_bytree': 0.6027952751910048}
# 0.9274762800150552
# 0.7863753705643247
# {'reg__random_state': 42, 'reg__n_estimators': 536, 'reg__max_depth': 7, 'reg__learning_rate': 0.058417768168456986,
#  'reg__gamma': 0.8893281747732737, 'reg__colsample_bytree': 0.5388503540959062}
# 0.9592934853879421
# 0.7921232303847281
# {'reg__random_state': 42, 'reg__n_estimators': 522, 'reg__max_depth': 8, 'reg__learning_rate': 0.08456187588526598,
#  'reg__gamma': 0.4128132943639499, 'reg__colsample_bytree': 0.48962008652863986}
# 0.9590479986844843
# 0.79488697635022
# {'reg__random_state': 42, 'reg__n_estimators': 604, 'reg__max_depth': 8, 'reg__learning_rate': 0.07137287160659607,
#  'reg__gamma': 0.42182404638273713, 'reg__colsample_bytree': 0.5178480668480605}
params = {
    # 'reg__colsample_bylevel': np.random.uniform(0.1, 1., n_iter).flatten(),
    'reg__colsample_bytree': np.random.uniform(0.2, 0.7, n_iter).flatten(),
    "reg__gamma": np.random.uniform(.3, .8, n_iter).flatten(),
    "reg__learning_rate": np.random.uniform(0.001, 0.5, n_iter).flatten(),  # default 0.1
    "reg__max_depth": np.random.randint(4, 9, n_iter).flatten(),  # default 3
    "reg__n_estimators": np.random.randint(400, 700, n_iter).flatten(),  # default 100
    # "reg__subsample": np.random.uniform(0.1, 1., n_iter).flatten(),
    "reg__random_state": [42]

}
print(time.ctime())
gs: RandomizedSearchCV = RandomizedSearchCV(estimator=pipe, n_iter=500, param_distributions=params,
                                            n_jobs=os.cpu_count() * 20 // 100, cv=3, verbose=3)
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
