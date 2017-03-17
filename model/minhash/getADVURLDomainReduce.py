#!/usr/bin/python
# -*- coding:utf8 -*-

import sys
import math
import random
import re

lastnetloc = None
sumcnt = 0
for line in sys.stdin :
	netloc = line.rstrip()

	if netloc != lastnetloc and lastnetloc != None :
		print lastnetloc, sumcnt
	
		sumcnt = 0

	lastnetloc = netloc	
	sumcnt += 1

print lastnetloc, sumcnt
