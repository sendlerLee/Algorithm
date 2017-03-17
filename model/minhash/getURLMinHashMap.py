#!/usr/bin/python
# -*- coding:utf8 -*-

import sys
import math
import random
import re
import urlparse
import urllib2
from urllib import unquote 

HashCnt = 50
for line in sys.stdin :
	line = line.replace("\t", "")
	line = line.rstrip().split("")
	if len(line) != 3 : continue

        url, pyid, cnt = line
	url = url.strip()
	url = unquote(url)

	netloc = None
	try :
		res = urlparse.urlparse(url)
		netloc = res.netloc
	except :
		continue
	if len(netloc) == 0 : continue

	hashvallist = []
	lastval = ""
	for i in range(HashCnt) :
		hashval = str(abs(hash(pyid + lastval)))
		lastval = hashval

		hashvallist.append(hashval)

	print netloc + "\t" + "\t".join(hashvallist)
