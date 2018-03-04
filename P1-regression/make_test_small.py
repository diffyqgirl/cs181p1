import csv
readfile = open("train.csv", 'rb')
writefile = open("train_small.csv", 'wb')
reader = csv.reader(readfile, delimiter=' ', quotechar='|')
writer = csv.writer(writefile, delimiter=' ', quotechar='|', quoting=csv.QUOTE_MINIMAL)
for i in range(1,10000):
	writer.writerow(reader.next())

readfile.close()
writefile.close()
readfile = open("train.csv", 'rb')
writefile = open("train2_small.csv", 'wb')
reader = csv.reader(readfile, delimiter=' ', quotechar='|')
writer = csv.writer(writefile, delimiter=' ', quotechar='|', quoting=csv.QUOTE_MINIMAL)
for i in range(10000,20000):
	writer.writerow(reader.next())

readfile.close()
writefile.close()