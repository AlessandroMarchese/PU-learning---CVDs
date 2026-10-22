from sklearn.impute import KNNImputer
import pandas as pd
import numpy as np

def knn_imputation(df):
    df["Cholesterol"] = df["Cholesterol"].replace(0, np.nan)
    df["RestingBP"] = df["RestingBP"].replace(0, np.nan)

    # transform string features to numeric
    sex_dict = {"M" : 0, "F": 1}
    df["Sex"] = df["Sex"].map(sex_dict)
    cpt_dict = {"TA" : 0, "ATA" : 1, "NAP" : 2, "ASY" : 3}
    df["ChestPainType"] = df["ChestPainType"].map(cpt_dict)
    restingecg_dict = {"Normal": 0, "ST" : 1, "LVH" : 2}
    df["RestingECG"] = df["RestingECG"].map(restingecg_dict)
    exangina_dict = {"N": 0, "Y": 1}
    df["ExerciseAngina"] = df["ExerciseAngina"].map(exangina_dict)
    stslope_dict = {"Up": 0, "Flat": 1, "Down": 2}
    df["ST_Slope"] = df["ST_Slope"].map(stslope_dict)


    imputer = KNNImputer(n_neighbors=3)
    imputed_df = imputer.fit_transform(df)
    chol_index = df.columns.get_loc("Cholesterol")
    df["Cholesterol"] = imputed_df[:, chol_index]
    chol_index = df.columns.get_loc("RestingBP")
    df["RestingBP"] = imputed_df[:, chol_index]
    df = df.astype({'Cholesterol': 'int32'})
    df = df.astype({'RestingBP': 'int32'})

    return df


if __name__ == "__main__":
    filename = "heart.csv"
    filename_imputed = "heart_processed.csv"
    df = pd.read_csv(filename)

    df = knn_imputation(df)

    df.to_csv(filename_imputed, sep=",")