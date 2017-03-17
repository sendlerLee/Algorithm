#!/usr/bin/python
# -*- coding:utf8 -*-

import sys
import math
import random
import re

lastpyid = None
urlset = set()
for line in sys.stdin :
	#print line,
	#continue
	line = line.rstrip().split("\t")
	if len(line) != 2 : continue
	pyid, url = line

	if pyid != lastpyid and lastpyid != None :

		if len(urlset) >= 2 :
			for cururl in urllset :
				print lastpyid + "\t" + cururl
			#print lastpyid + "\t" + "\t".join(list(urlset))

		urlset = set()


	if len(urlset) < 200 :
		urlset.add(url)

	lastpyid = pyid

#print lastpyid + "\t" + "\t".join(list(urlset))
