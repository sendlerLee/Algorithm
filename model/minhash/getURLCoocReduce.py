#!/usr/bin/python
# -*- coding:utf8 -*-

import sys
import math
import random
import re

cnt = 0
lasturlcooc = None
for line in sys.stdin :
	line = line.rstrip()
	urlcooc = line
	
	if urlcooc != lasturlcooc and lasturlcooc != None :
		print lasturlcooc, cnt

		cnt = 0

	lasturlcooc = urlcooc
	cnt += 1

print lasturlcooc, cnt
