import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal


data = np.array([[0,1] ,[1,2],[3,4],[5,2],[4,5],[8,5],[3,4],[1,2],[5,6]])

gauss = norm.fit(data)
mean = np.mean(data, axis=0)
cov = np.cov(data, rowvar=0)


var = multivariate_normal(mean=[0,0], cov=np.diag(np.diag(cov)))
var.pdf([1,0])


print(mean)
print(np.diag(np.diag(cov)))
print(var.pdf([1,0]))