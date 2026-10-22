# classic imports
import os

import numpy as np
import pandas as pd

# sklearn imports
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

# sarpu imports
from sarpu.biasedsampling import calc_probability_distribution
from sarpu.data_extending import generate_attributes
from sarpu.labeling_mechanisms import _calc_s


def artificial_attributes(x, y, n_extra_attributes, pdf_method=None):
    # generate the artificial attributes
    extra_atts = []

    prob_dist = calc_probability_distribution(x, y, pdf_method)

    for _ in range(n_extra_attributes):
        att_vals = generate_attributes(x, y, prob_dist=prob_dist, pdf_method=pdf_method)
        extra_atts.append(att_vals)

    return np.stack(extra_atts, axis=1)

def create_data_split(data_path, split):
    '''

    :param data_path:
    :param split:
    :return:
    '''
    # read data into dataframe
    df = pd.read_csv(data_path)

    # contstruct labels
    y = df['HeartDisease'].values

    # construct features
    df = df.drop(columns=["HeartDisease", df.columns[0]])
    x = df.to_numpy()

    # Normalize numerical data with MinMaxScaler
    scaler = MinMaxScaler()
    numerical_cols = ['Age', 'RestingBP', 'Cholesterol', 'MaxHR', 'Oldpeak']
    x[:, df.columns.get_indexer(numerical_cols)] = scaler.fit_transform(
        x[:, df.columns.get_indexer(numerical_cols)])

    # split into training and test set
    x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=split)

    return x_train, x_test, y_train, y_test

def add_artificial_attributes(x_train, x_test, y_train, y_test, labelling_model, pdf_method):
    '''
    :param x_train:
    :param x_test:
    :param y_train:
    :param y_test:
    :param labelling_model:
    :param pdf_mehod:
    :return:
    '''

    # construct original dataset via concatenation
    x = np.concatenate((x_train, x_test))
    y = np.concatenate((y_train, y_test))

    # add artificial attributes to end of x
    n_attributes = len(labelling_model.propensity_attributes)
    attr = artificial_attributes(x, y, n_attributes, pdf_method)
    x = np.concatenate((x, attr), axis=1)

    # resplit x_train and x_test
    x_train = x[:len(x_train), :]
    x_test  = x[len(x_train):, :]

    # Compute the propensity scores
    labelling_model.fit(x_train, y_train)
    propensity_scores = labelling_model.propensity_scores(x_train)

    # compute positive sample selection
    s_train = _calc_s(y_train, propensity_scores)

    return x_train, x_test, s_train

def write_dataset(x_train, x_test, y_train, y_test, s_train, data_path_folder):
    '''
        Write train.csv and test.csv to data_path
    :param x_train:
    :param x_test:
    :param y_train:
    :param y_test:
    :param s_train:
    :param data_path_folder:
    :return:
    '''

    # check if datapath folder exists, otherwise create
    if not os.path.exists(data_path_folder):
        # Create a new directory because it does not exist
        os.makedirs(data_path_folder)

    xys_train = np.concatenate((x_train, y_train[:, np.newaxis], s_train[:, np.newaxis]), axis=1)
    df_train = pd.DataFrame(xys_train)
    df_train.to_csv(path_or_buf=data_path_folder + "train.csv")

    xy_test = np.concatenate((x_test, y_test[:, np.newaxis]), axis=1)
    df_test = pd.DataFrame(xy_test)
    df_test.to_csv(path_or_buf=data_path_folder + "test.csv")

def read_dataset(path):
    '''
        Read train.csv and test.csv files
    :param path:
    :return:
    '''

    df_train = pd.read_csv(path + "train.csv")
    df_test = pd.read_csv(path + "test.csv")

    # drop index columns and convert to array
    df_train = df_train.drop(columns=[df_train.columns[0]])
    arr_train = df_train.to_numpy()
    x_train = arr_train[:, :15] # hardcoding number of features (with artificial attributes)
    y_train = arr_train[:, 15]
    s_train = arr_train[:, 16]

    df_test = df_test.drop(columns=[df_test.columns[0]])
    arr_test = df_test.to_numpy()
    x_test = arr_test[:, :15]
    y_test = arr_test[:, 15]

    return x_train, x_test, y_train, y_test, s_train