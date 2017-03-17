#!/usr/bin/python
# -*- coding:utf8 -*-

import sys
import math
import random
import re

HashCnt = 50
minhashlist = [sys.maxint] * HashCnt
lastnetloc = None
for line in sys.stdin :
	line = line.rstrip().split("\t")
	if len(line) != HashCnt + 1 : continue
	netloc, hashlist = line[0], line[1:]

	if netloc != lastnetloc and lastnetloc != None :
		print lastnetloc, "\t".join([str(x) for x in minhashlist])

		minhashlist = [sys.maxint] * HashCnt

	for i in range(len(hashlist)) :
		val = int(hashlist[i])
		if val < minhashlist[i] :
			minhashlist[i] = val
	lastnetloc = netloc

if netloc != lastnetloc and lastnetloc != None :
	print lastnetloc, "\t".join([str(x) for x in minhashlist])
