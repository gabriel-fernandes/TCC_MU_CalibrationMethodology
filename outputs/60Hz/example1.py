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
rcParams['figure.figsize'] = [7, 4]
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
    X = X/100
    freqs = fftpack.fftfreq(len(IAdataphase)) * f_s

    fig, ax = plt.subplots()

    ax.stem(freqs, np.abs(X))
    ax.set_xlabel('Frequência em Hertz [Hz]')
    #ax.set_ylabel('Frequency Domain (Spectrum) Magnitude')
    ax.set_ylabel('Espectro de Magnitude')
    ax.set_xlim(-f_s/2, f_s/2)
    ax.set_ylim(-5, 100)
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

def calcWorstRMS(vector, reference): 
    m = len(vector)
    error = np.empty(m, dtype=float)
    tmp = 0.0
    
    for idx in range(1,len(vector)):
        if vector[idx] > reference :
            tmp = vector[idx] - reference
        else:
            tmp = reference - vector[idx]
        error[idx] = tmp

   # print('errors', reference, vector)

    max_index = np.argmax(error, axis=0) 
    
 #   print('worst points for ref ', reference, ' : ', vector) 

    return vector[max_index]

def calcBestRMS(vector, reference): 
    m = len(vector)
    error = np.empty(m, dtype=float)
    tmp = 0.0
    
    for idx in range(1,len(vector)):
        if vector[idx] > reference :
            tmp = vector[idx] - reference
        else:
            tmp = reference - vector[idx]
        error[idx] = tmp

    #print('errors', reference, vector)
#    print('worst points for ref ', reference, ' : ', vector) 
    
    error = np.delete(error,0)

    min_index = np.argmin(error, axis=0) 
    
    
    return vector[min_index]

def calcAverageRMS(vector, reference): 
    m = len(vector)
    error = np.empty(m, dtype=float)
    tmp = 0.0
    
    for idx in range(1, len(vector)):
        tmp = tmp + vector[idx]
    
    tmp = tmp/(len(vector)-1)
    #print('errors', reference, vector)

    return tmp


def calcRMSChannel(vector, channel, isCurrent, calc = 0):
    ppc = 256
    tmp = 0.0
    sumRegister = np.empty(no_acq, dtype=float)
    sumRegister2 = np.empty(8, dtype=float)
    RMSRegister = np.empty(no_acq, dtype=float)
    outputRMS_vector = np.empty(no_acq, dtype=float)
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
     
    if calc == 0:
        #max error
        #RMS per cycle (256 ppc)
        if (isCurrent):
            for zdx in range(6):
                 outputRMS_vector[zdx] = calcWorstRMS(rms_matrix[zdx], currentValues[zdx]*currentFactor)
        else:
            for zdx in range(6):
                outputRMS_vector[zdx] = calcWorstRMS(rms_matrix[zdx], voltageValues[zdx]*voltageFactor)
    elif calc == 1:
        if (isCurrent):
            for zdx in range(6):
                 outputRMS_vector[zdx] = calcBestRMS(rms_matrix[zdx], currentValues[zdx]*currentFactor)
        else:
            for zdx in range(6):
                outputRMS_vector[zdx] = calcBestRMS(rms_matrix[zdx], voltageValues[zdx]*voltageFactor)
    elif calc == 2:
        #average error
        if (isCurrent):
            for zdx in range(6):
                 outputRMS_vector[zdx] = calcAverageRMS(rms_matrix[zdx], currentValues[zdx]*currentFactor)
        else:
            for zdx in range(6):
                outputRMS_vector[zdx] = calcAverageRMS(rms_matrix[zdx], voltageValues[zdx]*voltageFactor)

        
        
    return outputRMS_vector


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

IAdataphase = IAdata[5][:256]

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

#plt.plot(testx, testy, 'b')
#plt.xlabel("Amostras")
#plt.ylabel("Corrente (mA)")
#plt.plot(testx, IAdataphase, 'r')
#plt.show()

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

#getting min error
IAdataRMSmin = calcRMSChannel(IAdata, "IA", True, 1)/currentFactor
IBdataRMSmin = calcRMSChannel(IBdata, "IB", True, 1)/currentFactor
ICdataRMSmin = calcRMSChannel(ICdata, "IC", True, 1)/currentFactor
INdataRMSmin = calcRMSChannel(INdata, "IN", True, 1)/currentFactor
VAdataRMSmin = calcRMSChannel(VAdata, "VA", False, 1)/voltageFactor
VBdataRMSmin = calcRMSChannel(VBdata, "VB", False, 1)/voltageFactor
VCdataRMSmin = calcRMSChannel(VCdata, "VC", False, 1)/voltageFactor
VNdataRMSmin = calcRMSChannel(VNdata, "VN", False, 1)/voltageFactor

#gettin avg error
IAdataRMSavg = calcRMSChannel(IAdata, "IA", True, 2)/currentFactor
IBdataRMSavg = calcRMSChannel(IBdata, "IB", True, 2)/currentFactor
ICdataRMSavg = calcRMSChannel(ICdata, "IC", True, 2)/currentFactor
INdataRMSavg = calcRMSChannel(INdata, "IN", True, 2)/currentFactor
VAdataRMSavg = calcRMSChannel(VAdata, "VA", False, 2)/voltageFactor
VBdataRMSavg = calcRMSChannel(VBdata, "VB", False, 2)/voltageFactor
VCdataRMSavg = calcRMSChannel(VCdata, "VC", False, 2)/voltageFactor
VNdataRMSavg = calcRMSChannel(VNdata, "VN", False, 2)/voltageFactor


#getSignalFrequency(IAdataphase, 15360)

#print('IAdataRMS', IAdataRMS[0])
#print('currentVals', currentValues[0])

#getting worse RMS errors example
#IAdataRMSerrors = calcErrorRMS(IAdataRMS, currentValues)

#print('IAdataRMS before', IAdataRMS)

#begin algorithm evaluation
np.random.seed(123)

x = currentValues;
#x = voltageValues;
#x = [1, 2, 3, 4, 5, 6];
#y = [2.3, 4.3, 6.3 , 8.3, 10.3, 12.3];

"""
y = [IAdataRMS[0], IAdataRMS[1], IAdataRMS[2], IAdataRMS[3], IAdataRMS[4], IAdataRMS[5]]
print('Input sem calib max: ', y)


y = [IBdataRMS[0], IBdataRMS[1], IBdataRMS[2], IBdataRMS[3], IBdataRMS[4], IBdataRMS[5]]
print('Input sem calib max: ', y)
y = [ICdataRMS[0], ICdataRMS[1], ICdataRMS[2], ICdataRMS[3], ICdataRMS[4], ICdataRMS[5]]
print('Input sem calib max: ', y)
y = [INdataRMS[0], INdataRMS[1], INdataRMS[2], INdataRMS[3], INdataRMS[4], INdataRMS[5]]
print('Input sem calib: max ', y)

y = [IAdataRMSmin[0], IAdataRMSmin[1], IAdataRMSmin[2], IAdataRMSmin[3], IAdataRMSmin[4], IAdataRMSmin[5]]
print('Input sem calib: min ', y)

y = [IBdataRMSmin[0], IBdataRMSmin[1], IBdataRMSmin[2], IBdataRMSmin[3], IBdataRMSmin[4], IBdataRMSmin[5]]
print('Input sem calib: min ', y)
y = [ICdataRMSmin[0], ICdataRMSmin[1], ICdataRMSmin[2], ICdataRMSmin[3], ICdataRMSmin[4], ICdataRMSmin[5]]
print('Input sem calib: min ', y)
y = [INdataRMSmin[0], INdataRMSmin[1], INdataRMSmin[2], INdataRMSmin[3], INdataRMSmin[4], INdataRMSmin[5]]
print('Input sem calib: min ', y)


"""
y = [IAdataRMSavg[0], IAdataRMSavg[1], IAdataRMSavg[2], IAdataRMSavg[3], IAdataRMSavg[4], IAdataRMSavg[5]]
print('Input sem calib: avg ', y)
"""

y = [IBdataRMSavg[0], IBdataRMSavg[1], IBdataRMSavg[2], IBdataRMSavg[3], IBdataRMSavg[4], IBdataRMSavg[5]]
print('Input sem calib: avg ', y)
y = [ICdataRMSavg[0], ICdataRMSavg[1], ICdataRMSavg[2], ICdataRMSavg[3], ICdataRMSavg[4], ICdataRMSavg[5]]
print('Input sem calib: avg ', y)
y = [INdataRMSavg[0], INdataRMSavg[1], INdataRMSavg[2], INdataRMSavg[3], INdataRMSavg[4], INdataRMSavg[5]]
print('Input sem calib: avg ', y)

y = [VAdataRMS[0], VAdataRMS[1], VAdataRMS[2], VAdataRMS[3], VAdataRMS[4], VAdataRMS[5]]
print('Input sem calib max V: ', y)
y = [VBdataRMS[0], VBdataRMS[1], VBdataRMS[2], VBdataRMS[3], VBdataRMS[4], VBdataRMS[5]]
print('Input sem calib max V: ', y)

y = [VCdataRMS[0], VCdataRMS[1], VCdataRMS[2], VCdataRMS[3], VCdataRMS[4], VCdataRMS[5]]
print('Input sem calib max V: ', y)

y = [VNdataRMS[0], VNdataRMS[1], VNdataRMS[2], VNdataRMS[3], VNdataRMS[4], VNdataRMS[5]]
print('Input sem calib max V: ', y)

y = [VAdataRMSmin[0], VAdataRMSmin[1], VAdataRMSmin[2], VAdataRMSmin[3], VAdataRMSmin[4], VAdataRMSmin[5]]
print('Input sem calib: min V  ', y)
y = [VBdataRMSmin[0], VBdataRMSmin[1], VBdataRMSmin[2], VBdataRMSmin[3], VBdataRMSmin[4], VBdataRMSmin[5]]
print('Input sem calib: min V ', y)

y = [VCdataRMSmin[0], VCdataRMSmin[1], VCdataRMSmin[2], VCdataRMSmin[3], VCdataRMSmin[4], VCdataRMSmin[5]]
print('Input sem calib: min V ', y)

y = [VNdataRMSmin[0], VNdataRMSmin[1], VNdataRMSmin[2], VNdataRMSmin[3], VNdataRMSmin[4], VNdataRMSmin[5]]
print('Input sem calib: min V ', y)

y = [VAdataRMSavg[0], VAdataRMSavg[1], VAdataRMSavg[2], VAdataRMSavg[3], VAdataRMSavg[4], VAdataRMSavg[5]]
print('Input sem calib: avg V ', y)
y = [VBdataRMSavg[0], VBdataRMSavg[1], VBdataRMSavg[2], VBdataRMSavg[3], VBdataRMSavg[4], VBdataRMSavg[5]]
print('Input sem calib: avg V ', y)

y = [VCdataRMSavg[0], VCdataRMSavg[1], VCdataRMSavg[2], VCdataRMSavg[3], VCdataRMSavg[4], VCdataRMSavg[5]]
print('Input sem calib: avg V ', y)
y = [VNdataRMSavg[0], VNdataRMSavg[1], VNdataRMSavg[2], VNdataRMSavg[3], VNdataRMSavg[4], VNdataRMSavg[5]]
print('Input sem calib: avg V ', y)
"""



#y = [IAdataRMSavg[0], IAdataRMSavg[1], IAdataRMSavg[2], IAdataRMSavg[3], IAdataRMSavg[4], IAdataRMSavg[5]]

#print('Input sem calib: ', y)


#polyfit_res = np.polyfit(y, x, 2)


#polyfit_res = [-0.000035862743078,1.052297536959464, -9.121411258387482] #lad ordem3 sigy1

#polyfit_res = [-0.000008728139178, 0.927289675233447, 3.937766899317680] # lad ordem3 sig 1 1 1 0.5, 1 1]

#polyfit_res = [-0.000035375088158, 1.049619156527673, -6.943702907552121] #lad ordem 3 sig 1 1 1 0.5 0.5 1

#print('polyfit_res', polyfit_res)

#y_polyfit = [(polyfit_res[0]*IAdataRMS[0]**2) + IAdataRMS[0]*polyfit_res[1] + polyfit_res[2], polyfit_res[0]*IAdataRMS[1]**2 + polyfit_res[1]*IAdataRMS[1] + polyfit_res[2], polyfit_res[0]*IAdataRMS[2]**2 + polyfit_res[1]*IAdataRMS[2] + polyfit_res[2],  polyfit_res[0]*IAdataRMS[3]**2 + polyfit_res[1]*IAdataRMS[3] + polyfit_res[2],  polyfit_res[0]*IAdataRMS[4]**2 + polyfit_res[1]*IAdataRMS[4] + polyfit_res[2],  polyfit_res[0]*IAdataRMS[5]**2 + polyfit_res[1]*IAdataRMS[5] + polyfit_res[2]]


#print('polyfit:', y_polyfit)


#noise = 10 * np.random.normal(size=len(x))
#y = 10 * x + 10 + noise

#mask = np.arange(1, len(x)+1, 1) % 1.5 == 0
#print('mask ex', mask)

#y[mask] = np.linspace(6, 3, len(y[mask])) * y[mask]
#y[mask] = x*y[mask]

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

print('p', p)

#Correcting original wave


#IAdataNew  = np.empty(no_acq, dtype=object)

#for idx in range(no_acq):
#    IAdataNew[idx] = comtradeObjs[idx]['A'][0]['values']
#    IAdataNew[idx].pop()
    #IBdata[idx] = comtradeObjs[idx]['A'][1]['values']
    #IBdata[idx].pop()
    #ICdata[idx] = comtradeObjs[idx]['A'][2]['values']
    #ICdata[idx].pop()
    #INdata[idx] = comtradeObjs[idx]['A'][3]['values']
    #INdata[idx].pop()
    #VAdata[idx] = comtradeObjs[idx]['A'][4]['values']
    #VAdata[idx].pop()
    #VBdata[idx] = comtradeObjs[idx]['A'][5]['values']
    #VBdata[idx].pop()
    #VCdata[idx] = comtradeObjs[idx]['A'][6]['values']
    #VCdata[idx].pop()
    #VNdata[idx] = comtradeObjs[idx]['A'][7]['values']
    #VNdata[idx].pop()


#for kdx in range(len(IAdataNew[0])):
#    for jdx in range(no_acq):
#        IAdataNew[jdx][kdx] = (IAdataNew[jdx][kdx])/p[0] + np.abs(p[1])

#IAdataRMS = calcRMSChannel(IAdataNew, "IA", True)/currentFactor


#z = [IAdataRMS[0]/p[0] + p[1], IAdataRMS[1]/p[0] + p[1],IAdataRMS[2]/p[0] + p[1], IAdataRMS[3]/p[0] + p[1], IAdataRMS[4]/p[0] + p[1], IAdataRMS[5]/p[0] + p[1]]
#z = [IAdataRMSmin[0]/p[0] + p[1], IAdataRMSmin[1]/p[0] + p[1],IAdataRMSmin[2]/p[0] + p[1], IAdataRMSmin[3]/p[0] + p[1], IAdataRMSmin[4]/p[0] + p[1], IAdataRMSmin[5]/p[0] + p[1]]
z = [IAdataRMSavg[0]/p[0] + p[1], IAdataRMSavg[1]/p[0] + p[1],IAdataRMSavg[2]/p[0] + p[1], IAdataRMSavg[3]/p[0] + p[1], IAdataRMSavg[4]/p[0] + p[1], IAdataRMSavg[5]/p[0] + p[1]]
#z = [IBdataRMSavg[0]/p[0] + p[1], IBdataRMSavg[1]/p[0] + p[1],IBdataRMSavg[2]/p[0] + p[1], IBdataRMSavg[3]/p[0] + p[1], IBdataRMSavg[4]/p[0] + p[1], IBdataRMSavg[5]/p[0] + p[1]]
#z = [VCdataRMS[0]/p[0] + p[1], VCdataRMS[1]/p[0] + p[1],VCdataRMS[2]/p[0] + p[1], VCdataRMS[3]/p[0] + p[1], VCdataRMS[4]/p[0] + p[1], VCdataRMS[5]/p[0] + p[1]]
#z = [VCdataRMSmin[0]/p[0] + p[1], VCdataRMSmin[1]/p[0] + p[1],VCdataRMSmin[2]/p[0] + p[1], VCdataRMSmin[3]/p[0] + p[1], VCdataRMSmin[4]/p[0] + p[1], VCdataRMSmin[5]/p[0] + p[1]]
#z = [VCdataRMSavg[0]/p[0] + p[1], VCdataRMSavg[1]/p[0] + p[1],VCdataRMSavg[2]/p[0] + p[1], VCdataRMSavg[3]/p[0] + p[1], VCdataRMSavg[4]/p[0] + p[1], VCdataRMSavg[5]/p[0] + p[1]]


print('Aplicando ganhos e offsets: Minimos quadrados ord', z)

#print('new IAdataRMS', IAdataRMS)

#currentSinesN = np.empty(no_acq, dtype=object)
#IAdataphaseNew = IAdataRMS[0][:256]

#for jdx in range(no_acq):
#    currentSinesN[jdx] = referenceSine(currentValues[jdx]*1000, 256, 15360, 60)
 
#testx = np.arange(256)
#testy = currentSinesN[0]

#plt.plot(testx, testy, 'y')
#plt.plot(testx, IAdataphaseNew, 'r')
#plt.show()


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

#zN = [IAdataRMS[0]/m_l1 + b_l1, IAdataRMS[1]/m_l1 + b_l1, IAdataRMS[2]/m_l1 + b_l1, IAdataRMS[3]/m_l1 + b_l1, IAdataRMS[4]/m_l1 + b_l1, IAdataRMS[5]/m_l1 + b_l1]
#zN = [IAdataRMSmin[0]/m_l1 + b_l1, IAdataRMSmin[1]/m_l1 + b_l1,IAdataRMSmin[2]/m_l1 + b_l1, IAdataRMSmin[3]/m_l1 + b_l1, IAdataRMSmin[4]/m_l1 + b_l1, IAdataRMSmin[5]/m_l1 + b_l1]
zN = [IAdataRMSavg[0]/m_l1 + b_l1, IAdataRMSavg[1]/m_l1 + b_l1,IAdataRMSavg[2]/m_l1 + b_l1, IAdataRMSavg[3]/m_l1 + b_l1, IAdataRMSavg[4]/m_l1 + b_l1, IAdataRMSavg[5]/m_l1 + b_l1]
#zN = [IBdataRMSavg[0]/m_l1 + b_l1, IBdataRMSavg[1]/m_l1 + b_l1,IBdataRMSavg[2]/m_l1 + b_l1, IBdataRMSavg[3]/m_l1 + b_l1, IBdataRMSavg[4]/m_l1 + b_l1, IBdataRMSavg[5]/m_l1 + b_l1]

#zN = [VCdataRMS[0]/m_l1 + b_l1, VCdataRMS[1]/m_l1 + b_l1, VCdataRMS[2]/m_l1 + b_l1, VCdataRMS[3]/m_l1 + b_l1, VCdataRMS[4]/m_l1 + b_l1, VCdataRMS[5]/m_l1 + b_l1]
#zN = [VCdataRMSmin[0]/m_l1 + b_l1, VCdataRMSmin[1]/m_l1 + b_l1,VCdataRMSmin[2]/m_l1 + b_l1, VCdataRMSmin[3]/m_l1 + b_l1, VCdataRMSmin[4]/m_l1 + b_l1, VCdataRMSmin[5]/m_l1 + b_l1]
#zN = [VCdataRMSavg[0]/m_l1 + b_l1, VCdataRMSavg[1]/m_l1 + b_l1,VCdataRMSavg[2]/m_l1 + b_l1, VCdataRMSavg[3]/m_l1 + b_l1, VCdataRMSavg[4]/m_l1 + b_l1, VCdataRMSavg[5]/m_l1 + b_l1]


print('Aplicando ganhos e offsets LAD', zN)

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


