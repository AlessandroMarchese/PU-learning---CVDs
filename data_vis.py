import copy

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.impute import KNNImputer
from scipy.stats import pointbiserialr
def inspect_data():
    filename = "heart.csv"

    df = pd.read_csv(filename)

    # number of positives & negatives
    y =df["HeartDisease"].values
    c_pos = len(y[y==1])
    c_neg = len(y[y==0])
    print("P/N count: ", c_pos, "/", c_neg)

    ############
    # features #
    ############
    x = df.drop(columns="HeartDisease")
    print(x.describe())

    # feature 1: Age
    x_age = df["Age"].values

    # plot histogram of total and pos for each age
    fig, ax = plt.subplots()

    n, bins, patches = ax.hist(x_age, bins=range(0, 100, 5))
    n_p, bins_p, patches_p = ax.hist(x_age[y==1], bins=range(0, 100, 5))
    ax.set_xlabel("age")
    ax.set_title("Feature 1: Age")
    ax.legend(["total", "positive"])

    # feature 2: sex
    fig, ax = plt.subplots()
    total_sex_counts = df["Sex"].value_counts().values
    sub_df = df[df["HeartDisease"] == 1]
    pos_sex_counts = sub_df["Sex"].value_counts().values
    ticks = [1, 2]
    ax.bar(ticks, total_sex_counts)
    ax.bar(ticks, pos_sex_counts)
    ax.set_title("Feature 2: Sex")
    ax.set_xticks(ticks)
    ax.set_xticklabels(["M", "F"])
    plt.legend(["Total", "positive"])
    plt.show()

    # Feature 3: ChestPainType
    fig, ax = plt.subplots()
    total_CP_counts = df["ChestPainType"].value_counts().values
    pos_CP_counts = sub_df["ChestPainType"].value_counts().values
    ticks = [1, 2, 3, 4]
    ax.bar(ticks, total_CP_counts)
    ax.bar(ticks, pos_CP_counts)
    ax.set_title("Feature 3: ChestPainType")
    ax.set_xticks(ticks)
    ax.set_xticklabels(["TA", "ATA", "NAP", "ASY"])
    plt.legend(["Total", "positive"])
    plt.show()


    # Feature 4: RestingBP
    fig, ax = plt.subplots()
    x_restingBP = df["RestingBP"].values
    x_restingBP_pos = sub_df["RestingBP"].values

    ax.hist(x_restingBP, bins=range(75, np.max(x_restingBP), 10))
    ax.hist(x_restingBP_pos, bins=range(75, np.max(x_restingBP), 10))
    ax.set_title("Feature 4: RestingBP")
    ax.legend(["Total", "positive"])
    plt.show()

    # Feature 5: Cholesterol
    # before knn-imputation and after
    fig, ax = plt.subplots()
    x_chol = df["Cholesterol"].values
    x_chol_pos = sub_df["Cholesterol"].values
    ax.hist(x_chol, bins=range(0, np.max(x_chol), 10))
    ax.hist(x_chol_pos, bins=range(0, np.max(x_chol), 10))
    ax.set_title("Feature 5: Cholesterol")
    ax.legend(["Total", "positive"])
    plt.show()

    # feature 6: FastingBS
    fig, ax = plt.subplots()
    x_fastingbs = df["FastingBS"].value_counts().values
    x_fastingbs_pos = sub_df["FastingBS"].value_counts().values
    ticks = [1, 2]
    ax.bar(ticks, x_fastingbs)
    ax.bar(ticks, x_fastingbs_pos)
    ax.set_title("Feature 6: FastingBS")
    ax.set_xticks(ticks)
    ax.set_xticklabels(["< 120 mg/dl", "> 120 mg/dl"])
    plt.legend(["Total", "positive"])
    plt.show()

    # feature 7: RestingECG
    fig, ax = plt.subplots()
    x_restingecg = df["RestingECG"].value_counts().values
    x_restingecg_pos = sub_df["RestingECG"].value_counts().values
    ticks = [1, 2, 3]
    ax.bar(ticks, x_restingecg)
    ax.bar(ticks, x_restingecg_pos)
    ax.set_title("Feature 7: RestingECG")
    ax.set_xticks(ticks)
    ax.set_xticklabels(["Normal", "ST", "LVH"])
    plt.legend(["Total", "positive"])
    plt.show()

    # Feature 8: MaxHR
    fig, ax = plt.subplots()
    x_maxhr = df["MaxHR"].values
    x_maxhr_pos = sub_df["MaxHR"].values

    ax.hist(x_maxhr, bins=range(0, np.max(x_maxhr), 10))
    ax.hist(x_maxhr_pos, bins=range(0, np.max(x_maxhr), 10))
    ax.set_title("Feature 8: MaxHR")
    ax.legend(["Total", "positive"])
    plt.show()

    # Feature 9: ExerciseAngina
    fig, ax = plt.subplots()
    x_exerciseang = df["ExerciseAngina"].value_counts().values
    x_exerciseang_pos = sub_df["ExerciseAngina"].value_counts().values
    ticks = [1, 2]
    ax.bar(ticks, x_exerciseang)
    ax.bar(ticks, x_exerciseang_pos)
    ax.set_title("Feature 9: ExerciseAngina")
    ax.set_xticks(ticks)
    ax.set_xticklabels(["No", "Yes"])
    plt.legend(["Total", "positive"])
    plt.show()


    # Feature 10: Oldpeak
    fig, ax = plt.subplots()
    x_op = df["Oldpeak"].values
    x_op_pos = sub_df["Oldpeak"].values

    ax.hist(x_op, bins=range(0, int(np.max(x_op)), 1))
    ax.hist(x_op_pos, bins=range(0, int(np.max(x_op_pos)), 1))
    ax.set_title("Feature 10: Oldpeak")
    ax.legend(["Total", "positive"])
    plt.show()

    # Feature 11: ST_Slope
    fig, ax = plt.subplots()
    x_stslope = df["ST_Slope"].value_counts().values
    x_stslope_pos = sub_df["ST_Slope"].value_counts().values
    ticks = [1, 2, 3]
    ax.bar(ticks, x_stslope)
    ax.bar(ticks, x_stslope_pos)
    ax.set_title("Feature 11: ST_Slope")
    ax.set_xticks(ticks)
    ax.set_xticklabels(["Flat", "Up", "Down"])
    plt.legend(["Total", "positive"])
    plt.show()

    # combined feature: Male age
    x_male_age = df[df["Sex"]==0]["Age"].values
    x_male_age_pos = sub_df[sub_df["Sex"]==0]["Age"].values

    # plot histogram of total and pos
    fig, ax = plt.subplots()

    n, bins, patches = ax.hist(x_male_age, bins=range(0, 100, 5))
    n_p, bins_p, patches_p = ax.hist(x_male_age_pos, bins=range(0, 100, 5))
    ax.set_xlabel("age")
    ax.set_title("Male-Age")
    ax.legend(["total", "positive"])
    plt.show()

def knn_imputation(df):
    # correlation cholesterol & heartdisease before imputation
    pcorr, p_val = pointbiserialr(df["Cholesterol"].values, df["HeartDisease"].values)
    df["Cholesterol"] = df["Cholesterol"].replace(0, np.nan)

    # transform string features to numeric
    sex_dict = {"M" : 0, "F": 1}
    df["Sex"] = df["Sex"].map(sex_dict)
    cpt_dict = {"TA" : 0, "ATA" : 1, "NAP" : 2, "ASY" : 3}
    df["ChestPainType"] = df["ChestPainType"].map(cpt_dict)
    restingecg_dict = {"Normal": 0, "ST" : 1}
    df["RestingECG"] = df["RestingECG"].map(restingecg_dict)
    exangina_dict = {"N": 0, "Y": 1}
    df["ExerciseAngina"] = df["ExerciseAngina"].map(exangina_dict)
    stslope_dict = {"Up": 0, "Flat": 1, "Down": 2}
    df["ST_Slope"] = df["ST_Slope"].map(stslope_dict)


    imputer = KNNImputer(n_neighbors=3)
    imputed_df = imputer.fit_transform(df)
    chol_index = df.columns.get_loc("Cholesterol")
    df["Cholesterol"] = imputed_df[:, chol_index]

    # correlation cholesterol & heartdisease after imputation
    pcorr_after, p_val_after = pointbiserialr(df["Cholesterol"].values, df["HeartDisease"].values)

    print("Correlation cholesterol - CVD before imputation: ", pcorr, " with p-value: ", p_val)
    print("Correlation cholesterol - CVD after imputation: ", pcorr_after, " with p-value: ", p_val_after)

    return df

def biased_sampling_overlap():
    filename = "heart_processed.csv"

    df = pd.read_csv(filename)
    indicator1 = "Age"
    indicator2 = "MaxHR"
    age_max = np.max(df[indicator1].values)
    chol_max = np.max(df[indicator2].values)
    age_pos = df[df["HeartDisease"]==1][indicator1].values / age_max
    age_neg = df[df["HeartDisease"]==0][indicator1].values / age_max
    chol_pos = df[df["HeartDisease"]==1][indicator2] / chol_max
    chol_neg = df[df["HeartDisease"]==0][indicator2] / chol_max

    fig, ax = plt.subplots()
    ax.scatter(age_pos, chol_pos, c="b")
    ax.scatter(age_neg, chol_neg, c="r")
    plt.show()

def numerical_stats(df):
    num_cols = [0, 3, 4, 7, 9]
    df_num = df.iloc[: , num_cols]

    print(df_num.describe(include='all'))
if __name__ == '__main__':
    # let us look at the data first
    #inspect_data()
    #biased_sampling_overlap()

    df = pd.read_csv("heart.csv")
    #df = knn_imputation(df)
    numerical_stats(df)
