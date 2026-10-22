import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import normalize, MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

def plot_boundary(clf, x, y, feature_x, feature_y, df):
    # classifier predictions
    y_hat = clf.predict(x)

    # decision boundary for feature_x and feature_y
    # Retrieve the model parameters.
    b = clf.intercept_[0]
    w = clf.coef_.T
    w1 = w[feature_x]
    w2 = w[feature_y]
    # Calculate the intercept and gradient of the decision boundary.
    c = -b / w2
    m = -w1 / w2
    # calculate decision boundary line
    xmin, xmax = np.min(x[:, feature_x]), np.max(x[:, feature_x])
    ymin, ymax = np.min(x[:, feature_y]), np.max(x[:, feature_y])
    xd = np.array([xmin, xmax])
    yd = m * xd + c

    # plot
    fig, ax = plt.subplots()
    # plot decision boundary
    ax.plot(xd, yd, 'k', lw=1, ls='--')
    ax.fill_between(xd, yd, ymin, color='tab:blue', alpha=0.2)
    ax.fill_between(xd, yd, ymax, color='tab:orange', alpha=0.2)

    # plot samples
    x_TP = x[(y==1) & (y_hat==1)]
    x_FN = x[(y==1) & (y_hat==0)]
    x_TN = x[(y==0) & (y_hat==0)]
    x_FP = x[(y==0) & (y_hat==1)]
    ax.scatter(x_TP[:, feature_x], x_TP[:, feature_y], c='b', marker='o', s=8, alpha=0.5) # TP
    ax.scatter(x_FN[:, feature_x], x_FN[:, feature_y], c='b', marker='x', s=8, alpha=0.5) # FN
    ax.scatter(x_TN[:, feature_x], x_TN[:, feature_y], c='tab:orange', marker='o', s=8, alpha=0.5)  # TN
    ax.scatter(x_FP[:, feature_x], x_FP[:, feature_y], c='tab:orange', marker='x', s=8, alpha=0.5)  # FP
    ax.set_xlabel(df.columns[feature_x])
    ax.set_ylabel(df.columns[feature_y])
    plt.show()


if __name__ == "__main__":
    # read data into dataframe
    data_path = "heart_processed.csv"
    df = pd.read_csv(data_path)

    # contstruct labels
    y = df['HeartDisease'].values

    # construct features
    df = df.drop(columns=["HeartDisease", df.columns[0]])
    x = df.to_numpy()

    # Normalize numerical data
    scaler = MinMaxScaler()
    numerical_cols = ['Age', 'RestingBP', 'Cholesterol', 'MaxHR', 'Oldpeak']
    x[:, df.columns.get_indexer(numerical_cols)] = scaler.fit_transform(x[:, df.columns.get_indexer(numerical_cols)]-1)

    clf = LogisticRegression(random_state=331, max_iter=100).fit(x, y)
    score = clf.score(x, y)

    print("score LR: ", score)

    # try to find decision boundary on feature set that fails
    for i in range(len(x[0])-1):
        for j in range(i+1, len(x[0])):
            plot_boundary(clf, x, y, i, j, df)

    '''
        Accuracy is not 100 percent --> classes are not linearly separable
    '''