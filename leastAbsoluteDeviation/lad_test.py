'exec(%matplotlib inline)'
'exec(config IPython.matplotlib.backend = "retina")'

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import matplotlib.pyplot as plt
from matplotlib import rcParams
import scipy.optimize
#from matplotlib import latex

rcParams["figure.dpi"] = 300
rcParams["savefig.dpi"] = 300
rcParams['figure.figsize'] = [7, 5]
rcParams["text.usetex"] = True

import tensorflow as tf
tf.compat.v1.enable_eager_execution()
tf.executing_eagerly()
import numpy as np

np.random.seed(123)

x = np.linspace(1, 6, 6)
noise = 10 * np.random.normal(size=len(x))
y = 10 * x + 10 + noise

mask = np.arange(1, len(x)+1, 1) % 5 == 0
y[mask] = np.linspace(6, 3, len(y[mask])) * y[mask]

X = np.vstack([x, np.ones(len(x))])
tf.compat.v1.disable_v2_behavior()

# Ordinary Minimum Squares (using tensorflow for practicality)
with tf.compat.v1.Session() as sess:
    X_tensor = tf.convert_to_tensor(X.T, dtype=tf.float64)
    y_tensor = tf.reshape(tf.convert_to_tensor(y, dtype=tf.float64), (-1, 1))
    coeffs = tf.compat.v1.matrix_solve_ls(X_tensor, y_tensor)
    m, b = sess.run(coeffs)

#Coefficients from Minimum squares (without weighting)
p = [m[0], b[0]]

def sumdev(p):
    s = 0
    n = len(x)
    order = 1
    SIGY = [1, 1, 1, 1, 1, 1]
    for i in range(1, n):
        fx = 0
        for j in range(2):
            fx = fx + p[j]*x[i]**j
        s = s + abs(y[i] - fx)/SIGY[i]
    return s


s = sumdev(p)

print('s', s)

xopt = scipy.optimize.fmin(sumdev, p)

#coefficients from LAD
m_l1, b_l1 = xopt[0], xopt[1]

print('xopt', xopt)

plt.plot(x[~mask], y[~mask], 'ok', markersize=3., alpha=.5)
plt.plot(x[mask], y[mask], 'o', markersize=3., color='red', alpha=.5, label='Outliers')
plt.plot(x, 10 * x + 10, 'k', label="True line")
plt.plot(x, m * x + b, 'g', label="Least squares line")
plt.plot(x, m_l1 * x + b_l1, 'blue', label="Least absolute deviations line")
#plt.fill_between(x, (m_l1 - 1.96 * unc[0]) * x + b_l1 - 1.96 * unc[1],
#                    (m_l1 + 1.96 * unc[0]) * x + b_l1 + 1.96 * unc[1], alpha=.5)
plt.legend()
plt.show()


# Calculating R^2 Score OLS
ss_tot = 0
ss_res = 0
y_mean = np.mean(y)
for i in range(len(y)):
    y_pred = p[1] + p[0] * x[i]
    ss_tot += (y[i] - y_mean) ** 2
    ss_res += (y[i] - y_pred) ** 2
r2 = 1 - (ss_res/ss_tot)

print('R^2 OLS:', r2)

#getting R2 only from the points considered for LAD
y_filtered = y[~mask]

# Calculating R^2 Score LAD
ss_tot = 0
ss_res = 0
y_mean = np.mean(y_filtered)
for i in range(len(y_filtered)):
    y_pred = b_l1 + m_l1 * x[i]
    ss_tot += (y_filtered[i] - y_mean) ** 2
    ss_res += (y_filtered[i] - y_pred) ** 2
r2 = 1 - (ss_res/ss_tot)

print('R^2 LAD:', r2)



