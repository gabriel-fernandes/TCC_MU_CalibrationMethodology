#!/usr/bin/python3

import sys
import pyshark
import numpy
import struct
import csv
from comtradehandlers import writer

cap = pyshark.FileCapture(sys.argv[1]) 

nopackets=0

for idx, packet in enumerate(cap):
    nopackets+=1

print ('no of packets:', nopackets)

channels = [[0] * 9 for i in range(8*nopackets + 1)]
vallistStr = numpy.empty(8, dtype=object)
values_splitted = numpy.empty(64, dtype=object)
valueTemp = numpy.empty(8, dtype=int)

majorIndex = 0

#for idx in range(nopackets-1):
for idx in range(nopackets):
    vallist = list(cap[idx].sv.seqData.all_fields)

    for k in range(len(vallist)):
        vallistStr[k] = str(vallist[k])
        vallist[k] = vallistStr[k].replace("<LayerField sv.seqData: ","") 
        values_splitted = list(vallist[k].split(':'))

        for y in range(9):
            if y == 0:
                channels[majorIndex][y] = majorIndex
                majorIndex += 1
                continue
            valueTemp = struct.unpack('>i',bytes.fromhex(values_splitted[0 + 8*(y-1)] + values_splitted[1 + 8*(y-1)] + values_splitted[2 + 8*(y-1)] + values_splitted[3 + 8*(y-1)]))
            channels[majorIndex-1][y] = valueTemp[0]


headers = ["offset", "IA", "IB", "IC", "IN", "VA", "VB", "VC", "VN"]


with open('92_005.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(headers)
    writer.writerows(channels[:nopackets*8])
