#!/usr/bin/python
# -*- coding:utf8 -*-

import sys
import math
import random
import re

lastpyid = None
netloc2cnt = {}
for line in sys.stdin :
	line = line.rstrip().split("\t")
	if len(line) != 2 : continue
	pyid, netloc = line

	if pyid != lastpyid and lastpyid != None :
		print lastpyid + "" + "".join([x[0] + "" + str(x[1]) for x in sorted(netloc2cnt.items(), key = lambda x : x[1], reverse = True)])
	
		netloc2cnt = {}

	lastpyid = pyid
	
	netloc2cnt.setdefault(netloc, 0)
	netloc2cnt[netloc] += 1

print lastpyid + "" + "".join([x[0] + "" + str(x[1]) for x in sorted(netloc2cnt.items(), key = lambda x : x[1], reverse = True)])
