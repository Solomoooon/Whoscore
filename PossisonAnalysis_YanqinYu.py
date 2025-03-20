import pandas as pd
import statsmodels.api as sm
import glob
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error

# csv file paths for training and testing datasets
training_path = (
    "/Users/solomonyu/Desktop/SCU/Senior/Q2/CSCI 185/Project/Striker_csv/Training_Set"
)
testing_path = (
    "/Users/solomonyu/Desktop/SCU/Senior/Q2/CSCI 185/Project/Striker_csv/Testing_Set"
)

# load training csv
csv_training_files = glob.glob(os.path.join(training_path, "*.csv"))
df_training_list = []

for file in csv_training_files:
    df_temp = pd.read_csv(file)
    player_name = os.path.basename(file).split("_")[0].capitalize()
    df_temp["Player"] = player_name
    df_temp = df_temp[df_temp["Gls"].notna()]
    df_training_list.append(df_temp)


# combine training set
df_training_combined = pd.concat(df_training_list, ignore_index=True)

# only keep the following features in the csv
df_training_combined = df_training_combined[
    ["Player", "Season", "90s", "Gls", "Sh/90", "SoT/90", "xG", "Sh", "SoT", "npxG"]
]

df_training_combined[["90s", "Gls", "Sh/90", "SoT/90", "xG", "Sh", "SoT", "npxG"]] = (
    df_training_combined[
        ["90s", "Gls", "Sh/90", "SoT/90", "xG", "Sh", "SoT", "npxG"]
    ].apply(pd.to_numeric, errors="coerce")
)
df_training_clean = df_training_combined.dropna()

print("==== Cleaned training data frame ====")
print(df_training_clean.head())

# X_train is the list of features used in the model
y_train = df_training_clean["Gls"]
X_train = df_training_clean[["90s", "Sh/90", "SoT/90", "xG", "npxG"]]
X_train = sm.add_constant(X_train)

# use poisson distribution for training
poisson_model = sm.GLM(y_train, X_train, family=sm.families.Poisson())
poisson_results = poisson_model.fit()
print("\n==== Poisson Training Results ====")
print(poisson_results.summary())

df_training_clean["Predicted_Goals"] = poisson_results.predict(X_train)

print(df_training_clean[["Player", "Season", "Gls", "Predicted_Goals"]])

# load csv for testing dataset
csv_testing_files = glob.glob(os.path.join(testing_path, "*.csv"))

df_testing_list = []
for file in csv_testing_files:
    df_temp = pd.read_csv(file)
    player_name = os.path.basename(file).split("_")[0].capitalize()
    df_temp["Player"] = player_name
    df_temp_clean = df_temp[df_temp["Gls"].notna()]
    df_testing_list.append(df_temp_clean)

# combine csvs
df_testing_combined = pd.concat(df_testing_list, ignore_index=True)

df_testing_combined = df_testing_combined[
    ["Player", "Season", "90s", "Gls", "Sh/90", "SoT/90", "xG", "Sh", "SoT", "npxG"]
]

df_testing_combined[["90s", "Gls", "Sh/90", "SoT/90", "xG", "Sh", "SoT", "npxG"]] = (
    df_testing_combined[
        ["90s", "Gls", "Sh/90", "SoT/90", "xG", "Sh", "SoT", "npxG"]
    ].apply(pd.to_numeric, errors="coerce")
)
df_testing_clean = df_testing_combined.dropna()

print("\n==== Cleaned testing data frame  ====")
print(df_testing_clean.head())

# y_test and x features
y_test = df_testing_clean["Gls"]
X_test = df_testing_clean[["90s", "Sh/90", "SoT/90", "xG", "npxG"]]
X_test = sm.add_constant(X_test)

# predict using trained model
df_testing_clean["Predicted_Goals"] = poisson_results.predict(X_test)

# output
print("\n==== Prediction results ====")
print(df_training_clean[["Player", "Season", "Gls", "Predicted_Goals"]])

# Variance for training set
training_variance = df_training_clean["Gls"].var()
training_std = df_training_clean["Gls"].std()

# Variance for testing set
testing_variance = df_testing_clean["Gls"].var()
testing_std = df_testing_clean["Gls"].std()

print("===== Training Set =====")
print(f"Variance: {training_variance:.2f}")
print(f"Standard Deviation: {training_std:.2f}")

print("\n===== Testing Set =====")
print(f"Variance: {testing_variance:.2f}")
print(f"Standard Deviation: {testing_std:.2f}")

# MAE_training
training_mae = mean_absolute_error(
    df_training_clean["Gls"], df_training_clean["Predicted_Goals"]
)

# MAE_testing
testing_mae = mean_absolute_error(
    df_testing_clean["Gls"], df_testing_clean["Predicted_Goals"]
)

print("\n===== MAE =====")
print(f"Training Set MAE: {training_mae:.2f}")
print(f"Testing Set MAE: {testing_mae:.2f}")


fig, axes = plt.subplots(2, 1, figsize=(12, 10))

axes[0].plot(
    df_training_clean["Player"] + " (" + df_training_clean["Season"] + ")",
    df_training_clean["Gls"],
    marker="o",
    linestyle="-",
    linewidth=2,
    markersize=6,
    label="Actual Goals (y)",
)

axes[0].plot(
    df_training_clean["Player"] + " (" + df_training_clean["Season"] + ")",
    df_training_clean["Predicted_Goals"],
    marker="x",
    linestyle="--",
    linewidth=2,
    markersize=6,
    label="Predicted Goals (ŷ)",
)

axes[0].set_title("Actual vs Predicted Goals (Training Set)", fontsize=16)
axes[0].set_xlabel("Player (Season)", fontsize=14)
axes[0].set_ylabel("Goals", fontsize=14)
axes[0].tick_params(axis="x", rotation=45)
axes[0].legend()
axes[0].grid(True, linestyle="--", alpha=0.6)

axes[1].plot(
    df_testing_clean["Player"] + " (" + df_testing_clean["Season"] + ")",
    df_testing_clean["Gls"],
    marker="o",
    linestyle="-",
    linewidth=2,
    markersize=6,
    label="Actual Goals (y)",
)

axes[1].plot(
    df_testing_clean["Player"] + " (" + df_testing_clean["Season"] + ")",
    df_testing_clean["Predicted_Goals"],
    marker="x",
    linestyle="--",
    linewidth=2,
    markersize=6,
    label="Predicted Goals (ŷ)",
)

axes[1].set_title("Actual vs Predicted Goals (Testing Set)", fontsize=16)
axes[1].set_xlabel("Player (Season)", fontsize=14)
axes[1].set_ylabel("Goals", fontsize=14)
axes[1].tick_params(axis="x", rotation=45)
axes[1].legend()
axes[1].grid(True, linestyle="--", alpha=0.6)

plt.tight_layout()

plt.figure(figsize=(12, 6))

sns.histplot(
    df_training_clean["Gls"],
    kde=True,
    bins=10,
    label="Training Set",
    color="blue",
    alpha=0.5,
)
sns.histplot(
    df_testing_clean["Gls"],
    kde=True,
    bins=10,
    label="Testing Set",
    color="orange",
    alpha=0.5,
)

plt.title("Goals Distribution: Training vs Testing Set")
plt.xlabel("Goals")
plt.ylabel("Frequency")
plt.legend()

plt.show()
