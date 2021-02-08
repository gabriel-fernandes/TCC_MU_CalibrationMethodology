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

channels = [[0] * 9 for i in range(nopackets)]

#for idx in range(nopackets-1):
for idx in range(nopackets):
    vallist = list(cap[idx].layers[1]._all_fields.values())
    values_filtered = vallist[12]
    values_splitted = list(values_filtered.split(':'))
    for y in range(9):
        if y == 0:
            channels[idx][y] = idx
            continue
        valueTemp = struct.unpack('>i',bytes.fromhex(values_splitted[0 + 8*(y-1)] + values_splitted[1 + 8*(y-1)] + values_splitted[2 + 8*(y-1)] + values_splitted[3 + 8*(y-1)]))
        channels[idx][y] = valueTemp[0] 

headers = ["offset", "IA", "IB", "IC", "IN", "VA", "VB", "VC", "VN"]


with open('template.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(headers)
    writer.writerows(channels[:nopackets])
