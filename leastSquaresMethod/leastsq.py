from sklearn.datasets import make_regression
from matplotlib import pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression

#least squares linear

#building random test vector
X, y, coefficients = make_regression(
    n_samples=6,
    n_features=1,
    n_informative=1,
    n_targets=1,
    noise=5,
    coef=True,
    random_state=1
)

x_uf =  np.transpose(X)
x = x_uf[0,:]

X_mean = np.mean(X)
y_mean = np.mean(y)

n = len(y)

# Using the formula to calculate 'm' and 'c'
numer = 0
denom = 0
for i in range(n):
    numer += (X[i] - X_mean) * (y[i] - y_mean)
    denom += (X[i] - X_mean) ** 2

#slope m
m = numer / denom

#intercept c
c = y_mean - (m * X_mean)
 
# Printing coefficients
print("Coefficients:")
print('m:', m, 'c:', c)


# Calculating R^2 Score
ss_tot = 0
ss_res = 0
for i in range(n):
    y_pred = c + m * X[i]
    ss_tot += (y[i] - y_mean) ** 2
    ss_res += (y[i] - y_pred) ** 2
r2 = 1 - (ss_res/ss_tot)

print("R2 Score:", r2)


#building plot

x_graph = np.linspace(min(x), max(x), 2)
y_graph = x_graph*m + c


title  = "Graph of {gain}*x + {offset}"
title_f = (title.format(gain = m, offset = c))
print(title_f)

plt.scatter(X, y)
plt.plot(x_graph, y_graph, c='black')
plt.title(title_f)
plt.show()
