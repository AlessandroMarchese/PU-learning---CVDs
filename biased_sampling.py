import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN, KMeans
from scipy.stats import multivariate_normal
from sklearn.neighbors import KernelDensity
from sklearn.preprocessing import normalize
from random import choices


def biasedSampV1(df):
    pos_age = df[df["HeartDisease"]==1]["Age"].values
    pos_age = pos_age/np.max(pos_age)
    pos_sex = df[df["HeartDisease"]==1]["Sex"].values
    pos_chol = df[df["HeartDisease"]==1]["Cholesterol"].values
    pos_chol = pos_chol/np.max(pos_chol)
    pos_hr = df[df["HeartDisease"] == 1]["MaxHR"].values
    #X = np.transpose(np.stack((pos_age, pos_sex, pos_chol)))
    X = np.transpose(np.stack((pos_age, pos_chol)))

    '''
        AGE - CHOLESTEROL colored with sex
    '''
    fig, ax = plt.subplots()
    # male scatter plot
    ax.scatter(pos_age[pos_sex==0], pos_chol[pos_sex==0], c='k')
    # female scatter plot
    ax.scatter(pos_age[pos_sex==1], pos_chol[pos_sex==1], c='y')

    ax.set_xlabel("age")
    ax.set_ylabel("cholesterol")
    ax.set_title("age-chol-sex")
    ax.legend(["male", "female"])

    plt.show()

    '''
        Test density-based clustering: DBSCAN
    '''
    eps = 0.02
    minPts = 10
    db = DBSCAN(eps=eps, min_samples=minPts).fit(X)

    labels = db.labels_

    # Number of clusters in labels, ignoring noise if present.
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)

    print("Number of clusters found: ", n_clusters_)
    print("Number of noise samples: ", n_noise_)

    # plot clusters
    fig = plt.figure()
    ax = fig.add_subplot()
    for i in range(n_clusters_):
        ax.scatter(pos_age[labels == i], pos_chol[labels == i])

    # 3D plot with age
    #ax = fig.add_subplot(projection="3d")
    #ax.scatter(pos_age[labels == 0], pos_chol[labels == 0], pos_sex[labels == 0], c='b')
    #ax.scatter(pos_age[labels == 1], pos_chol[labels == 1], pos_sex[labels == 1], c='r')

    plt.show()


    '''
       KMEANS 
    '''
    kmeans = KMeans(n_clusters=2).fit_predict(X)

    fig, ax = plt.subplots()
    ax.scatter(X[:,0], X[:,1], c=kmeans)
    ax.set_xlabel("Age")
    ax.set_ylabel("Cholesterol")
    ax.set_title("Kmeans")
    plt.show()

    ''' 
        generate distribution of 3D points (age, sex, cholesterol)
    '''


    '''
        multinomial distribution
    '''
    #calculate mean of each feature
    #calculate variance of each feature
    # create diagonal covariance matrix (assumes independent variables)
    # size: (res_f1, res_f2, res_f3)

    mean = np.mean(X, axis=0)
    print(mean)
    cov = np.cov(X, rowvar=0)
    var = multivariate_normal(mean=mean, cov=np.diag(np.diag(cov)))
    probs = var.pdf(X)

    sorted_indices = np.argsort(probs)
    quantity_of_undefined = int(len(X)*0.70)
    undefined_indices = sorted_indices[:quantity_of_undefined]
    c = []
    for i in range(len(X)):
        if i in undefined_indices:
            c.append("black")
        else:
            c.append("white")


    fig, ax = plt.subplots()
    heat_X = var.rvs(100000)
    plt.hist2d(heat_X[:, 0], heat_X[:, 1], bins=100, cmap="plasma")

    ax.scatter(X[:, 0], X[:, 1], s = 4, c=c)
    ax.set_xlabel("Age")
    ax.set_ylabel("Cholesterol")
    ax.set_title("Multivariate_normal")

    plt.show()

    '''
            multinomial distribution with probs
    '''
    # calculate mean of each feature
    # calculate variance of each feature
    # create diagonal covariance matrix (assumes independent variables)
    # size: (res_f1, res_f2, res_f3)

    mean = np.mean(X, axis=0)
    print(mean)
    cov = np.cov(X, rowvar=0)
    ## var = multivariate_normal(mean=mean, cov=np.diag(np.diag(cov)))
    ## probs = var.pdf(X)
    probs = np.array(multivariate_normal.pdf(X,mean=mean, cov = np.diag(np.diag(cov))))
    print(np.average(probs))
    probs = probs/ np.average(probs)
    print(probs)
    s_labels = []
    for prob in probs:
        choice = choices([0,1], weights=[1-prob, prob])[0]
        s_labels.append(choice)

    c = []
    for label in s_labels:
        if label == 0:
            c.append("black")
        else:
            c.append("white")

    fig, ax = plt.subplots()
    heat_X = var.rvs(100000)
    plt.hist2d(heat_X[:, 0], heat_X[:, 1], bins=100, cmap="plasma")

    ax.scatter(X[:, 0], X[:, 1], s=4, c=c)
    ax.set_xlabel("Age")
    ax.set_ylabel("Cholesterol")
    ax.set_title("Multivariate_normal")

    plt.show()

    '''
                multinomial distribution with probs
    '''


    '''
        z = normalized(attr1) + normalized(attr2) + ...
    '''
    z_scores = []
    for x in X:
        z_score = 0
        for attr in x:
            z_score += attr
        z_scores.append(z_score)

    sorted_indices = np.argsort(z_scores)
    quantity_of_undefined = int(len(X) * 0.70)
    undefined_indices = sorted_indices[:quantity_of_undefined]
    c = []
    for i in range(len(X)):
        if i in undefined_indices:
            c.append("black")
        else:
            c.append("green")

    fig, ax = plt.subplots()
    ax.scatter(X[:, 0], X[:, 1], c=c)
    ax.set_xlabel("Age")
    ax.set_ylabel("Cholesterol")
    ax.set_title("z(attr)")

    plt.show()

def kernelDensityEstimation(df):
    '''
        kernel density estimation:
            https://scikit-learn.org/stable/modules/density.html#kernel-density

    '''
    # Test plot (1D)
    pos_age = df[df["HeartDisease"] == 1]["Age"].values
    pos_age = pos_age / np.max(pos_age)
    pos_sex = df[df["HeartDisease"] == 1]["Sex"].values
    pos_chol = df[df["HeartDisease"] == 1]["Cholesterol"].values
    pos_chol = pos_chol
    pos_hr = df[df["HeartDisease"] == 1]["MaxHR"].values

    X = np.transpose(np.stack((pos_age, pos_sex)))
    bw = 0.2
    # try out different kernels: Gaussian, Tophat, epanechnkov, exponential
    kde_gauss = KernelDensity(kernel='gaussian', bandwidth=bw).fit(X)
    kde_tophat = KernelDensity(kernel='tophat', bandwidth=bw).fit(X)
    kde_epanechnikov = KernelDensity(kernel='epanechnikov', bandwidth=bw).fit(X)
    kde_exponential = KernelDensity(kernel='exponential', bandwidth=bw).fit(X)
    # scores on age distribution
    n_points = 1000
    x_plot = np.transpose(np.stack((np.linspace(0, 1, n_points), np.zeros((n_points,)))))
    gauss_scores = kde_gauss.score_samples(x_plot)
    tophat_scores = kde_tophat.score_samples(x_plot)
    epanechnikov_scores = kde_epanechnikov.score_samples(x_plot)
    exponential_scores = kde_exponential.score_samples(x_plot)
    # show results on feature 1: Age
    fig, ax = plt.subplots()

    ax.hist(X[:, 0], bins=np.linspace(0, 1, 7), fc="#AAAAFF", density=True, stacked=True)
    ax.plot(x_plot[:, 0], np.exp(gauss_scores), c='r')
    ax.plot(x_plot[:, 0], np.exp(tophat_scores), c='k')
    ax.plot(x_plot[:, 0], np.exp(epanechnikov_scores), c='b')
    ax.plot(x_plot[:, 0], np.exp(exponential_scores), c='g')

    plt.show()

    # Fit kernel to cholesterol and age
    pos_age = df[df["HeartDisease"] == 1]["Age"].values
    pos_age = pos_age / np.max(pos_age)
    pos_chol = df[df["HeartDisease"] == 1]["Cholesterol"].values
    pos_chol = pos_chol / np.max(pos_chol)
    X = np.transpose(np.stack((pos_age, pos_chol)))
    bw = 0.1
    kde = KernelDensity(kernel='epanechnikov', bandwidth=bw).fit(X)

    # plot contour
    n_points = 100
    xgrid = np.linspace(0, 1, n_points)
    ygrid = np.linspace(0, 1, n_points)
    X, Y = np.meshgrid(xgrid, ygrid)

    xy = np.vstack([X.ravel(), Y.ravel()]).T
    Z = kde.score_samples(xy)
    Z = Z.reshape(X.shape)

    plt.contourf(X, Y, Z, cmap=plt.cm.coolwarm)

    plt.show()


if __name__ == "__main__":

    # read processed dataset
    df = pd.read_csv("heart_processed.csv")

    biasedSampV1(df)

