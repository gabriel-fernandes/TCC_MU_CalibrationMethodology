import numpy as np 
from sklearn.datasets import make_regression
import matplotlib.pyplot as plt 
plt.style.use('ggplot') 
# generate 15 observations of x on the interval [1,50] 
#x = np.linspace(1, 4, 6) 

#building random test vector
Xrdm, yrdm, coefficients = make_regression(
    n_samples=6,
    n_features=1,
    n_informative=1,
    n_targets=1,
    noise=5,
    coef=True,
    random_state=1
)

Xrdm = np.abs(Xrdm)

x_uf =  np.transpose(Xrdm)
x = x_uf[0,:]
#x = [50, 200, 1000, 1200, 3000, 4000]
x = [9.2, 57.0, 92.0, 115.0, 138.0, 161.0]
x = np.transpose(x)

# create random noise with mean 0 and standard deviation 2 
noise = np.random.normal(0, 2, 6) 
# make a heteroskedastic model to fit 
#yh = 3.1 + 2.5*x + x*noise
#yh = [49.6972991542454, 198.488597917192, 992.544072907594, 1191.33889083726, 2977.74061381542, 3970.69335122444]
#yh = [50.0724848415868, 198.836173238584, 996.366556277949, 1195.83240657591, 2986.44670260655, 3984.91919247198]
#yh = [50.0724848415868, 198.836173238584, 996.366556277949, 1195.83240657591, 2986.44670260655, 3984.91919247198]
#yh = [49.9492270870285, 198.655447332845, 994.490926748931, 1193.9885646907, 2980.78563119532, 3976.25377295255]
#yh = [9.0364600309847, 56.5918151365535, 90.5460609982967, 113.211976441429,  135.828305011585, 158.467404620755 ]
#yh = [9.10376781095908, 56.6885620422123, 90.8937040501754, 113.629733612429, 136.221531207814, 159.031729671052 ]
yh = [9.07207788157551, 56.6270805920401, 90.7219114324304, 113.461977423264, 135.961607811372, 157.628146649472 ]
# make a model constant variance 
#y = 3.1 + 2.5*x + noise
#y = [49.6972991542454, 198.488597917192, 992.544072907594, 1191.33889083726, 2977.74061381542, 3970.69335122444]
#y = [50.0724848415868, 198.836173238584, 996.366556277949, 1195.83240657591, 2986.44670260655, 3984.91919247198]
#y = [49.9492270870285, 198.655447332845, 994.490926748931, 1193.9885646907, 2980.78563119532, 3976.25377295255]
#y = [49.9492270870285, 198.655447332845, 994.490926748931, 1193.9885646907, 2980.78563119532, 3976.25377295255]

#y = [9.0364600309847, 56.5918151365535, 90.5460609982967, 113.211976441429,  135.828305011585, 158.467404620755 ]
#y = [9.10376781095908, 56.6885620422123, 90.8937040501754, 113.629733612429, 136.221531207814, 159.031729671052 ]
y = [9.07207788157551, 56.6270805920401, 90.7219114324304, 113.461977423264, 135.961607811372, 157.628146649472 ]


#y = 3.1 + 2.5*x + noise 
# use OLS and WLS to fit both models 
# start with OLS case 
X = np.matrix([np.ones(6), x]).T # data matrix
Y = np.dot(X.T, np.array(y).reshape(-1, 1)) # forces vector 
H = np.dot(X.T, X) 
b = np.linalg.solve(H, Y)
print('sol OLS:', b)
# now solve the WLS case 
# build a weights matrix with wi = 1/xi 
#W = np.diag([1/xi for xi in x]) 
#W = np.diag([1/0.06, 1/0.02, 1/0.01, 1/0.01, 1/0.01, 1/0.04]) 
W = np.diag([1, 1, 1, 1, 1, 1]) 
w_diag = W.diagonal()
HH = np.dot(np.dot(X.T, W), X) 
YH = np.dot(np.dot(X.T, W), np.array(y).reshape(-1,1)) 
bh = np.linalg.solve(HH, YH)
print('sol WLS:', bh)
# difference parameter estimates to see how much they differ 
print("Difference b - bh = ({:.3f}, {:.3f})".format(b[0,0] - bh[0,0], b[1,0] - bh[1,0])) 

# Calculating R^2 Score for Ordinary Least Squares
ss_tot = 0
ss_res = 0
y_mean = np.mean(y)
print('ymean', y_mean)
for i in range(6):
    y_pred = 3.1 + 2.5 * x[i]
    ss_tot += (y[i] - y_mean) ** 2
    ss_res += (y[i] - y_pred) ** 2
r2 = 1 - (ss_res/ss_tot)

print("R2 Score OLS:", r2)

# Calculating R^2 Score for Weighted Least Squares
ss_tot = 0
ss_res = 0
y_wmean = np.average(y, axis=0, weights=w_diag)
print('weighted', y_wmean)
for i in range(6):
    y_pred = 3.1 + 2.5 * x[i]
    ss_tot += w_diag[i]*(y[i] - y_wmean) ** 2
    ss_res += w_diag[i]*(y[i] - y_pred) ** 2
r2 = 1 - (ss_res/ss_tot)

print("R2 Score WLS:", r2)

# replot data with regression lines included 
# black dots are heteroskedastic noise 
# blue dots are constant variance 
y_sol = [y[0]/bh[1,0] + bh[0,], y[1]/bh[1,0] + bh[0,], y[2]/bh[1,0] + bh[0,], y[3]/bh[1,0] + bh[0,], y[4]/bh[1,0] + bh[0,], y[5]/bh[1,0] + bh[0,]]

print('sol ', y_sol)

plt.figure(figsize=(10,5)) 
plt.plot(x, yh, 'c.', alpha=0.5, label='nonconstant variance') 
plt.plot(x, bh[0,0] + bh[1,0]*x, 'c', alpha=0.75, label='WLS line') 
plt.plot(x, y, 'm.', alpha=0.5, label='constant variance') 
plt.plot(x, b[0,0] + b[1,0]*x, 'm', alpha=0.75, label='OLS line') 
plt.legend() 
plt.show()

