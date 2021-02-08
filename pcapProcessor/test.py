import csv

row_list = [ ["offset", "IA", "IB", "IC", "IN", "VA", "VB", "VC", "VN"],
             [1, "Linus Torvalds", "Linux Kernel"],
             [2, "Tim Berners-Lee", "World Wide Web"],
             [3, "Guido van Rossum", "Python Programming"]]

#data_list = ["offset", "IA", "IB", "IC", "IN", "VA", "VB", "VC", "VN"]
with open('template.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerows(row_list)
