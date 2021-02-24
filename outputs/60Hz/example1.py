#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2013 Miguel Moreto <http://sites.google.com/site/miguelmoreto/>

# This file is part of pyComtrade.
#
#    pyComtrade is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    any later version.
#
#    pyComtrade is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with pyComtrade.  If not, see <http://www.gnu.org/licenses/>.
# ====================================================================
#
# This is an example of using the pyComtrade module to read a comtrade record.
# The Comtrade data are in the test_data folder.
#
# Developed by Miguel Moreto
# Brazil - 2013
#
__version__ = "$Revision$"  # SVN revision.
__date__ = "$Date$"         # Date of the last SVN revision.

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

# Matplotlib module is needed for this example.
# pyComtrade needs numpy.
import pyComtrade
import numpy
from matplotlib import pylab
from pylab import *

no_acq = 6
no_ch = 8
voltageFactor = 100000
currentFactor = 1000

def calcRMSChannel(vector):
    ppc = 256
    tmp = 0.0
    sumRegister = np.empty(no_acq, dtype=float)
    sumRegister2 = np.empty(49, dtype=float)
    RMSRegister = np.empty(no_acq, dtype=float)
    RMSRegister2 = np.empty(49, dtype=float)
    
    i=0
    for kdx in range(no_acq):
        for ydx in range(len(vector[kdx]) - 1):
            sumRegister[kdx] += vector[kdx][ydx]**2
            #if ((ydx % ppc) == 0):
            #    sumRegister2[i] = tmp
            #    tmp = 0.0
            #    i+=1
    

   # for zdx in range(i):
   #     RMSRegister2[zdx] = np.sqrt(sumRegister2[zdx]/(i))

#    print ('RMS Reg2', RMSRegister2)

    for jdx in range(no_acq):
        RMSRegister[jdx] = np.sqrt(sumRegister[jdx]/(len(vector[jdx]) - 1))

    return RMSRegister


# Create an instance of the ComtradeRecord class and read the CFG file:
comtradeObj = pyComtrade.ComtradeRecord()


comtradeObjs = np.empty(no_acq, dtype=object)

comtradeObjs[0] = pyComtrade.ComtradeRecord();
comtradeObj.read('./92_005.cfg', './92_005.dat')


for idx in range (6):
    comtradeObjs[idx] = pyComtrade.ComtradeRecord()


comtradeObjs[0].read('./92_005.cfg', './92_005.dat')
comtradeObjs[1].read('./57_02.cfg', './57_02.dat')
comtradeObjs[2].read('./92_1.cfg', './92_1.dat')
comtradeObjs[3].read('./115_12.cfg', './115_12.dat')
comtradeObjs[4].read('./138_3.cfg', './138_3.dat')
comtradeObjs[5].read('./161_4.cfg', './161_4.dat')

print(comtradeObj.get_analog_ids())  # print the ids of the analog channels.

N = comtradeObj['endsamp'][-1]

print('Record has {} samples'.format(N))
print('Sampling rate is {} samples/sec.'.format(comtradeObj['samp'][-1]))

# Reading channel 4:
AnalogChannelData = comtradeObj['A'][0]['values']

#DigitalChannelData = comtradeObj['D'][1]['values']

# Reading time vector:
time = comtradeObj.get_timestamps()


Nn = np.empty(no_acq, dtype=int)
time_n = np.empty(no_acq, dtype=object)

for idx in range (no_acq):
    print(comtradeObjs[idx].get_analog_ids())  # print the ids of the analog channels.

    Nn[idx] = comtradeObjs[idx]['endsamp'][-1]

    print('Record has {} samples'.format(Nn[idx]))
    print('Sampling rate is {} samples/sec.'.format(comtradeObjs[idx]['samp'][-1]))
    
    time_n[idx] = comtradeObjs[idx].get_timestamps();
    print('timestamps')
    print(len(time_n[idx]))
    # Reading channel 4:


IAdata  = np.empty(no_acq, dtype=object)
IBdata  = np.empty(no_acq, dtype=object)
ICdata  = np.empty(no_acq, dtype=object)
INdata  = np.empty(no_acq, dtype=object)
VAdata  = np.empty(no_acq, dtype=object)
VBdata  = np.empty(no_acq, dtype=object)
VCdata  = np.empty(no_acq, dtype=object)
VNdata  = np.empty(no_acq, dtype=object)

for idx in range(no_acq):
    IAdata[idx] = comtradeObjs[idx]['A'][0]['values']
    IBdata[idx] = comtradeObjs[idx]['A'][1]['values']
    ICdata[idx] = comtradeObjs[idx]['A'][2]['values']
    INdata[idx] = comtradeObjs[idx]['A'][3]['values']
    VAdata[idx] = comtradeObjs[idx]['A'][4]['values']
    VBdata[idx] = comtradeObjs[idx]['A'][5]['values']
    VCdata[idx] = comtradeObjs[idx]['A'][6]['values']
    VNdata[idx] = comtradeObjs[idx]['A'][7]['values']


# Reading time vector:
time = comtradeObj.get_timestamps()

IAdataRMS = calcRMSChannel(IAdata)/currentFactor
IBdataRMS = calcRMSChannel(IBdata)/currentFactor
ICdataRMS = calcRMSChannel(ICdata)/currentFactor
INdataRMS = calcRMSChannel(INdata)/currentFactor
VAdataRMS = calcRMSChannel(VAdata)/voltageFactor
VBdataRMS = calcRMSChannel(VBdata)/voltageFactor
VCdataRMS = calcRMSChannel(VCdata)/voltageFactor
VNdataRMS = calcRMSChannel(VNdata)/voltageFactor

print('RMS IA 50mA', IAdataRMS[0])
print('RMS IA 200mA', IAdataRMS[1])
print('RMS VA 9.2V', VAdataRMS[0])
print('RMS VA 57V', VAdataRMS[1])

#begin test VA
np.random.seed(123)

x = [9.2, 57.0, 92.0, 115.0, 138.0, 161.0];
y = [VAdataRMS[0], VAdataRMS[1], VAdataRMS[2], VAdataRMS[3], VAdataRMS[4], VAdataRMS[5]]

#noise = 10 * np.random.normal(size=len(x))
#y = 10 * x + 10 + noise

#mask = np.arange(1, len(x)+1, 1) % 5 == 0
#y[mask] = np.linspace(6, 3, len(y[mask])) * y[mask]

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

print('p ', p)

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
m_l1, b_l1 = xopt[1], xopt[0]

m_l1 = np.asarray(m_l1)
b_l1 = np.asarray(b_l1)

print('xopt', xopt)

print('m b l1', m_l1, b_l1)

plt.plot(x, y, 'ok', markersize=3., alpha=.5)
#plt.plot(x[mask], y[mask], 'o', markersize=3., color='red', alpha=.5, label='Outliers')
#plt.plot(x, 10 * x + 10, 'k', label="True line")
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
#y_filtered = y[~mask]
y_filtered = y

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


