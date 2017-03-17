#!/usr/bin/python
# -*- coding:utf8 -*-

import sys
import math
import random
import re
import urlparse

CampanyID = 965

whitlist = {}
for line in file("whitelist") :
	url = line.lstrip()
	whitlist[url] = 1

cheatingurldict = {}
for line in file("cheating_domain_20150414") :
	line = line.lstrip().split()
	cheatingurl = line[1]
	score = float(line[-1])
	label = line[0]
	if label == "1" : continue
	if score >= 1.0 : cheatingurldict[cheatingurl] = 1

for line in file("url_cooc_20150324_result") :
	oldline = line
	line = line.lstrip().split()
	if len(line) != 5 : continue
	url, other_url, cnt, cooc_cnt, score = line
	if url in cheatingurldict and float(score) >= 5 :
		print oldline,
