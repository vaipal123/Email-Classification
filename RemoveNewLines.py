# coding: utf-8
from bs4 import BeautifulSoup
import os
import codecs
import io
import re	
from string import digits
for filename in os.listdir('spam'):	
	clean = open('spam/' + filename).read().replace('\n', ' ') # remove all new lines in file 
	#clean = re.sub(r'''(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:'".,<>?“”‘’]))''', '' , clean)
	clean = re.sub(r'\w+:\/{2}[\d\w-]+(\.[\d\w-]+)*(?:(?:\/[^\s/]*))*', ' ', clean) #USE THIS FOR REMOVING URL
	clean = re.sub(r'([^\s\w]|_)+', ' ', clean)
	clean = re.sub(r'([a-z])([A-Z])', r'\1 \2', clean)
	clean = ''.join([i for i in clean if not i.isdigit()]) #removing digits
	clean = ' '.join(clean.split())  # remove extra spaces
 	with open('spam/' + filename, "w") as f:
		f.write(clean)
# str = 'this is some text that will have one form or the other url embeded, most will have valid URLs while there are cases where they can be bad. for eg, http://www.google.com and http://www.google.co.uk and www.domain.co.uk and etc.'
# print res
