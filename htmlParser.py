from bs4 import BeautifulSoup
import os
import codecs
import io
def remove_non_ascii_1(text):
	return ''.join([i if ord(i) < 128 else ' ' for i in text])
for filename in os.listdir('ham'):	
	with open('ham/' + filename) as infile:
	    nonascii = bytearray(range(0x80, 0x100))
	    with open('d_parsed.txt','wb') as outfile:
    		for line in infile: # b'\n'-separated lines (Linux, OSX, Windows)
        		outfile.write(line.translate(None, nonascii))
	with open('d_parsed.txt','r') as markup:
	    soup = BeautifulSoup(markup.read())

	with open('ham/' + filename, "w") as f:
		q = soup.get_text()
		# content = unicode(q.content.strip(codecs.BOM_UTF8), 'utf-8')
		# parser.parse(StringIO.StringIO(content))
		p = remove_non_ascii_1(q)
		f.write(p)

