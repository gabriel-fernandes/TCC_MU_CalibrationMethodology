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
from matplotlib import pylab
from pylab import *
import matplotlib.pyplot as plt

no_acq = 6
no_ch = 8
voltageFactor = 100000
currentFactor = 1000
currentValues = [50.0, 200.0, 1000.0, 1200.0, 3000.0, 4000.0]
voltageValues = [9.2, 57.0, 92.0, 115.0, 138.0, 161.0]


def getSignalFrequency(vector, f_s):
    from scipy import fftpack

    X = fftpack.fft(vector)
    freqs = fftpack.fftfreq(len(IAdataphase)) * f_s

    fig, ax = plt.subplots()

    ax.stem(freqs, np.abs(X))
    ax.set_xlabel('Frequency in Hertz [Hz]')
    ax.set_ylabel('Frequency Domain (Spectrum) Magnitude')
    ax.set_xlim(-f_s / 2, f_s / 2)
    ax.set_ylim(-5, 10000)
    plt.show()


def referenceSine(refval, no_samples, f_acq, freq):
    x = np.arange(no_samples)
    y = np.sqrt(2)*refval*np.sin(2 * np.pi * freq * x/f_acq)


#    plt.plot(x, y)
#    plt.xlabel('sample(n)')
#    plt.ylabel('voltage(V)')
#    plt.show()

    return y

def referenceCosine(refval, no_samples, f_acq, freq):
    x = np.arange(no_samples)
    y = np.sqrt(2)*refval*np.cos(2 * np.pi *freq* x/f_acq)

    return y

def calcErrorRMS(vector, reference):
    m = 61
    absError = np.empty(m, dtype=float)
    tmp = 0.0
    
    for idx in range(len(vector)):
        if vector[idx] > reference[idx] :
            tmp = vector[idx] - reference[idx]
        else:
            tmp = reference[idx] - vector[idx]
        absError[idx] = tmp
 
    #print('errors', absError)
    return absError


def calcWorstRMS(vector, reference): 
    m = 60
    m = m+1
    error = np.empty(m, dtype=float)
    tmp = 0.0
    
    for idx in range(len(vector)):
        if vector[idx] > reference :
            tmp = vector[idx] - reference
        else:
            tmp = reference - vector[idx]
        error[idx] = tmp

    max_index = np.argmax(error, axis=0) 
    
    return vector[max_index]

def calcRMSChannel(vector, channel, isCurrent):
    ppc = 256
    tmp = 0.0
    sumRegister = np.empty(no_acq, dtype=float)
    sumRegister2 = np.empty(8, dtype=float)
    RMSRegister = np.empty(no_acq, dtype=float)
    worstRMS_vector = np.empty(no_acq, dtype=float)
    errors_vector = np.empty(no_acq, dtype=float)
    RMSRegister2 = np.empty(8, dtype=float)
    
    i=0
    
    m = int((len(vector[0]))/ppc)
    m = m+1
    n = no_acq

    accum_matrix = [[0 for x in range(m)] for y in range(no_acq)]
    rms_matrix = [[0 for x in range(m)] for y in range(no_acq)]

    for kdx in range(no_acq):
        for ydx in range(len(vector[kdx])):
            sumRegister[kdx] += vector[kdx][ydx]**2
            tmp  += vector[kdx][ydx]**2
            if ((ydx % (ppc-1)) == 0):
                if(ydx == 0):
                    continue
                accum_matrix[kdx][i] = tmp
                tmp = 0.0
                i+=1
        i = 0
   

    for jdx in range(no_acq):
        for ydx in range(m):
            rms_matrix[jdx][ydx] = np.sqrt(accum_matrix[jdx][ydx]/(ppc))
     
    #RMS per cycle (256 ppc)
    if (isCurrent):
        for zdx in range(6):
             worstRMS_vector[zdx] = calcWorstRMS(rms_matrix[zdx], currentValues[zdx]*currentFactor)
    else :
        for zdx in range(6):
            worstRMS_vector[zdx] = calcWorstRMS(rms_matrix[zdx], voltageValues[zdx]*voltageFactor)

    #print('worst rms', worstRMS_vector)

   # RMS for all cycles
   # for jdx in range(no_acq):
   #     for ydx in range (m):
   #         RMSRegister[jdx] = np.sqrt(sumRegister[jdx]/(len(vector[jdx]) - 1))
    print('worstRMS', worstRMS_vector)
    return worstRMS_vector


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


#creating ideal sine waves
currentSines = np.empty(no_acq, dtype=object)
currentCosines = np.empty(no_acq, dtype=object)
voltageSines = np.empty(no_acq, dtype=object)
voltageCosines = np.empty(no_acq, dtype=object)



for jdx in range(no_acq):
    currentSines[jdx] = referenceSine(currentValues[jdx], 256, 15360, 60)
    currentCosines[jdx] = referenceCosine(currentValues[jdx], 256, 15360, 60)
    voltageSines[jdx] = referenceSine(voltageValues[jdx], 256, 15360, 60)
    voltageCosines[jdx] = referenceCosine(voltageValues[jdx], 256, 15360, 60)

IAdata  = np.empty(no_acq, dtype=object)
IBdata  = np.empty(no_acq, dtype=object)
ICdata  = np.empty(no_acq, dtype=object)
INdata  = np.empty(no_acq, dtype=object)
VAdata  = np.empty(no_acq, dtype=object)
VBdata  = np.empty(no_acq, dtype=object)
VCdata  = np.empty(no_acq, dtype=object)
VNdata  = np.empty(no_acq, dtype=object)

#getting signals from comtrade, popping last one (maybe it's a bug)
for idx in range(no_acq):
    IAdata[idx] = comtradeObjs[idx]['A'][0]['values']
    IAdata[idx].pop()
    IBdata[idx] = comtradeObjs[idx]['A'][1]['values']
    IBdata[idx].pop()
    ICdata[idx] = comtradeObjs[idx]['A'][2]['values']
    ICdata[idx].pop()
    INdata[idx] = comtradeObjs[idx]['A'][3]['values']
    INdata[idx].pop()
    VAdata[idx] = comtradeObjs[idx]['A'][4]['values']
    VAdata[idx].pop()
    VBdata[idx] = comtradeObjs[idx]['A'][5]['values']
    VBdata[idx].pop()
    VCdata[idx] = comtradeObjs[idx]['A'][6]['values']
    VCdata[idx].pop()
    VNdata[idx] = comtradeObjs[idx]['A'][7]['values']
    VNdata[idx].pop()


import scipy.fftpack

IAdataphase = IAdata[0][:256]

IAdataphase = [IAdataphase / currentFactor for IAdataphase in IAdataphase]


innerSin = np.inner(currentSines[0], np.array(IAdataphase))
#innerSin = np.inner(currentSines[0], currentSines[0])
innerCosin = np.inner(currentCosines[0], np.array(IAdataphase))
#innerCosin = np.inner(currentCosines[0], currentCosines[0])
print('inner sin', innerSin)
#print('inner sin', np.inner(np.array(IAdata[0]), currentSines[0]))
print('inner cosin', innerCosin)

phaseDisplacement = np.arctan(innerSin/innerCosin)

print('Phase disp in deg:', phaseDisplacement*(360/np.pi))

testx = np.arange(256)
testy = currentSines[0]

plt.plot(testx, testy, 'b')
plt.plot(testx, IAdataphase, 'r')
plt.show()

#getting frequency

#print('inner sin', np.inner(np.array(IAdata[0]), currentSines[0]))
#print('inner sin', np.dot(currentSines[0], currentCosines[0]))
#print('inner cos', np.dot(currentCosines[0], currentSines[0]))
#print('inner cosin', np.inner(np.array(IAdata[0]), currentCosines[0]))

# Reading time vector:
time = comtradeObj.get_timestamps()

#getting worse RMS
IAdataRMS = calcRMSChannel(IAdata, "IA", True)/currentFactor
IBdataRMS = calcRMSChannel(IBdata, "IB", True)/currentFactor
ICdataRMS = calcRMSChannel(ICdata, "IC", True)/currentFactor
INdataRMS = calcRMSChannel(INdata, "IN", True)/currentFactor
VAdataRMS = calcRMSChannel(VAdata, "VA", False)/voltageFactor
VBdataRMS = calcRMSChannel(VBdata, "VB", False)/voltageFactor
VCdataRMS = calcRMSChannel(VCdata, "VC", False)/voltageFactor
VNdataRMS = calcRMSChannel(VNdata, "VN", False)/voltageFactor

print('IAdataRMS', IAdataRMS[0])
print('currentVals', currentValues[0])

#getting worse RMS errors example
IAdataRMSerrors = calcErrorRMS(IAdataRMS, currentValues)

print('IAdataRMSerrors', IAdataRMSerrors)

#begin algorithm evaluation
np.random.seed(123)

x = currentValues;
y = [IAdataRMS[0], IAdataRMS[1], IAdataRMS[2], IAdataRMS[3], IAdataRMS[4], IAdataRMS[5]]

#noise = 10 * np.random.normal(size=len(x))
#y = 10 * x + 10 + noise

mask = np.arange(1, len(x)+1, 1) % 5 == 0
#y[mask] = np.linspace(6, 3, len(y[mask])) * y[mask]
#y[mask] = currentValues * y[mask]

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


