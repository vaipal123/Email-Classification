import os
import random

with open('finalResultSpam.txt') as f:
	a = f.readlines()
	b = [x for x in a if x != 'spam\t\n']

# b is the final list of all spam entries

with open('finalResultHam.txt') as g:
	c = g.readlines()
	d = [x for x in c if x != 'ham\t\n']

e = map(next, random.sample([iter(b)]*len(b) + [iter(d)]*len(d), len(b)+len(d)))
ret = ''.join(e)
with open('finalDataSet.txt', 'w') as final:
	final.write(ret)



