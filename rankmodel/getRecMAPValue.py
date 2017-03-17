#!/usr/bin/python
# -*- coding:utf8 -*-

import sys
import math
import random
import re
import urlparse

traindate = 20150605

user2itemlabeldict = {}
filename = "campany_1095_%s_test_feat" % (str(traindate))
for line in file(filename) :
        line = line.rstrip().split()
	useritem, label = line[:2]
	user, item = useritem.split(":")
	user2itemlabeldict.setdefault(user, {})
	user2itemlabeldict[user][item] = int(label)

user2itemscoreldict = {}
filename = "campaign_clk.plr.result.%s" %(str(traindate))
for line in file(filename) :
        line = line.rstrip().split()
	useritem, score = line
	user, item = useritem.split(":")

	user2itemscoreldict.setdefault(user, {})
	user2itemscoreldict[user][item] = float(score)
	user2itemscoreldict[user][item] = random.random()
	user2itemscoreldict[user][item] = 0


#print "Load data finish\n"

predictdate = 20150605
filename = "campany_1095_pv_clk_%s_10_days" % (str(predictdate))
for line in file(filename) :
	line = line.rstrip().split()
	if len(line) != 6 : continue
	pyid, cvtstr, clkstr, pvstr, advstr, bidunbidstr = line
	if pyid not in user2itemscoreldict : continue

	advlist = advstr.split("")[1:]
	for item in advlist :
		item = item.split("")
		if len(item) < 2 : continue
		actiontime, itemid = item[0], item[1]
		if itemid in user2itemscoreldict[pyid] and int(actiontime[:8]) < predictdate : 
			user2itemscoreldict[pyid][itemid] = 0 - int(actiontime)
			user2itemscoreldict[pyid][itemid] = int(actiontime)
				
top1match = 0
sumscore = 0
totalscore = 0
totalrec = 0
weightscore = 0
for user in user2itemscoreldict :
	item2score = user2itemscoreldict[user]
	item2label = user2itemlabeldict[user]

	idx = 1
	matchidx = 1
	for item in sorted(item2score.items(), key = lambda x : x[1], reverse = True) :
		item, score = item
		label = item2label[item]
		if label == 1 :
			sumscore += idx
			matchidx = idx
		if idx == 1 and label == 1 :
			top1match += 1
			
		idx += 1
	totalscore += len(item2score)
	weightscore += matchidx * 1.0 / len(item2score)
	totalrec += 1

avgreccnt = totalscore * 1.0 / totalrec
algreccnt = sumscore * 1.0 / totalrec

print avgreccnt, algreccnt
print weightscore, weightscore / totalrec
print top1match, totalrec, top1match * 1.0 / totalrec
