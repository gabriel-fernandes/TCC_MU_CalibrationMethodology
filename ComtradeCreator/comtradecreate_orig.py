from comtradehandlers import writer
import csv
import datetime
# lets use now as the start time
start_time = datetime.datetime.now()

# ...and 20 mills later as the trigger time
trigger_time = start_time + datetime.timedelta(milliseconds=20)

comtradeWriter = writer.ComtradeWriter("test_orig.cfg", start_time, trigger_time,rec_dev_id=250)

#created_id = comtradeWriter.add_digital_channel("RELAY1", 0, 0, 0)
#print("Created new digital channel " + str(created_id))

#created_id = comtradeWriter.add_digital_channel("RELAY2", 0, 0, 0)
#print("Created new digital channel " + str(created_id))

# A Current
created_id = comtradeWriter.add_analog_channel("IA", "A", "I", uu="A", skew=0, min=-500, max=500, primary=1,
                                               secondary=1)
print("Created new analog channel " + str(created_id))

# B Current
created_id = comtradeWriter.add_analog_channel("IB", "B", "I", uu="A", skew=0, min=-500, max=500, primary=1,
                                               secondary=1)
print("Created new analog channel " + str(created_id))

# C Current
created_id = comtradeWriter.add_analog_channel("IC", "C", "I", uu="A", skew=0, min=-500, max=500, primary=1,
                                               secondary=1)
print("Created new analog channel " + str(created_id))

# open a CSV file with raw data to insert. column 0 contains a microsecond offset from start_time
with open('./examples/outdata.csv', 'r') as csvfile:
    datareader = csv.reader(csvfile, delimiter=',', quotechar='|')
    next(datareader, None)  # skip the header
    for row in datareader:
        comtradeWriter.add_sample_record(row[0],
                                         [row[1], row[2], row[3]]
                                        )

comtradeWriter.finalize()
