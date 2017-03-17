#!/usr/bin/python
# -*- coding:utf8 -*-

import sys
import math
import random
import re
import urlparse
from datetime import *

node2pos = {}
node2cnt = {}
node2score = {}
filename = "campaign_ext.test.result"
for line in file(filename) :
	line = line.rstrip().split()
	if len(line) != 4 : continue
	pyid, label, score, node = line
	node = int(node)

	if int(label) == 1 :
		node2pos.setdefault(node, 0)
		node2pos[node] += 1

	node2cnt.setdefault(node, 0)
	node2cnt[node] += 1

	
	node2score[node] = score

for item in sorted(node2score.items(), key = lambda x : x[1], reverse = True) :
	node, score = item
	pos = 0
	if node in node2pos : pos = node2pos[node]
	cnt = node2cnt[node]

	print node, score, pos, cnt, pos * 1.0 / cnt
