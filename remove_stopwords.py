from nltk.corpus import stopwords
import numpy as np

stop = set(stopwords.words('english'))

sentence = "this is a foo bar sentence"
TRAIN_SET_PATH = "SpamCollection.txt"

X, y = [], []
with open(TRAIN_SET_PATH, "rb") as infile:
    for line in infile:
    	z = []
        label, text = line.split("\t")
        for i in text.lower().split():
	       	if i not in stop:
		        z.append(i)  
 
        X.append(z)
        y.append(label)
X, y = np.array(X), np.array(y)

print X[0];
print X[1];
print X[2];
print X[3];

X, y = [], []
with open(TRAIN_SET_PATH, "rb") as infile:
    for line in infile:
        label, text = line.split("\t")
 
        X.append(text.split())
        y.append(label)
X, y = np.array(X), np.array(y)
print "total examples %s" % len(y)


print X[0];
print X[1];
print X[2];
print X[3];

