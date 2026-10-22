from matplotlib import pyplot as plt
from scipy.stats import multivariate_normal
import pandas as pd
from sarpu.data_generator import read_dataset
import numpy as np

x_train, x_test, y_train, y_test, s_train = read_dataset("../dataset/")

#df[df["HeartDisease"]==1]["Age"].values

age_values = x_train[:,0]
cholesterol_values = x_train[:,4]

age_values_pos = []
cholesterol_values_pos = []
colors = []
i = 0
while i in range(len(y_train)):
    if y_train[i] == 1:
        age_values_pos.append(age_values[i])
        cholesterol_values_pos.append(cholesterol_values[i])
        if s_train[i] == 1:
            colors.append("white")
        else:
            colors.append("black")
    i += 1

age_values_pos = np.array(age_values_pos)
cholesterol_values_pos = np.array(cholesterol_values_pos)
print(age_values_pos)
print(cholesterol_values_pos)
X = np.concatenate((age_values_pos[:, np.newaxis], cholesterol_values_pos[:, np.newaxis]), axis = 1)
mean = np.mean(X, axis=0)
cov = np.cov(X, rowvar=0)
var = multivariate_normal(mean=mean, cov=np.diag(np.diag(cov)))


fig, ax = plt.subplots()
heat_X = var.rvs(100000)
plt.hist2d(heat_X[:, 0], heat_X[:, 1], bins=100, cmap="plasma", )
ax.scatter(age_values_pos, cholesterol_values_pos, s=5, c = colors)
ax.scatter([],[],s=5, c= ["black"])
ax.set_xlabel("Age")
ax.set_ylabel("Cholesterol")
ax.set_title("Multinomial labeling mechanism result")
plt.legend(["positive", "undefined"], loc ="lower right")

plt.show()





