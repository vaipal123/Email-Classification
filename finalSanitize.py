import os
import codecs
import io
import re

res = ''
for filename in os.listdir('spam'):
	inputString = open('spam/' + filename).read()
	res += 'spam' + '\t' + inputString.lower() + '\n'

with open('finalResultSpam.txt' , "a") as g:
	g.write(res)	