__author__ = 'sam.royston'
import csv
import numpy as np
import sys
from matplotlib import pyplot as plt

epochs = []
train_perp = []
valid_perp = []
wps = []
dw_norm = []
time_taken = []

def read_file(file):
    with open(file, mode='r') as f:
        return [row for row in csv.reader(f)]

def parse_row(row):
    if 'epoch' in row[0]:
        parse_training(row)
    elif 'Validation' in row[0]:
        parse_validation(row[0].split(':'))

def parse_validation(valid):
    validation_error = valid[1].replace('\t','').replace(' ','')
    valid_perp.append(float(validation_error))


def parse_training(training):
    epoch = training[0].split('=')[1].replace(' ','')
    epochs.append(float(epoch))

    training_perplexity = training[1].split('=')[1].replace(' ','')
    train_perp.append(float(training_perplexity))

    words_ps = training[2].split('=')[1].replace(' ','')
    wps.append(float(words_ps))

    norm = training[3].split('=')[1].replace(' ','')
    dw_norm.append(float(norm))

    time_sb = training[4].split('=')[1].replace(' ','')
    time_taken.append(float(time_sb))

def parse_rows(rows):
    for row in rows:
        parse_row(row)


raw = read_file(sys.argv[1])
parse_rows(raw)

X1 = np.array(epochs)[11:]
X2 = X1[::10]

Y1 = np.array(train_perp[11:])
print len(np.array(valid_perp)), len(X2)
Y2 = np.interp(X1, X2,np.array(valid_perp[0:-1]))

plt.xlabel("Epochs")
plt.ylabel("Perplexity")

plt.plot(X1,Y1)
plt.plot(X1,Y2)
plt.show()

print train_perp
print valid_perp