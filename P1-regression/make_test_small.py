import csv
readfile = open("train.csv", 'rb')
writefile = open("train_small.csv", 'wb')
reader = csv.reader(readfile, delimiter=' ', quotechar='|')
writer = csv.writer(writefile, delimiter=' ', quotechar='|', quoting=csv.QUOTE_MINIMAL)
for i in range(1,10000):
	writer.writerow(reader.next())

readfile.close()
writefile.close()
readfile = open("test.csv", 'rb')
writefile = open("test_small.csv", 'wb')
reader = csv.reader(readfile, delimiter=' ', quotechar='|')
writer = csv.writer(writefile, delimiter=' ', quotechar='|', quoting=csv.QUOTE_MINIMAL)
for i in range(1,10000):
	writer.writerow(reader.next())

readfile.close()
writefile.close()