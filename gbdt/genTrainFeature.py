#!/usr/bin/python
# -*- coding:utf8 -*-

import sys
import math
import random
import re
import urlparse
from datetime import *

IdAdvertiserCompanyId = "1631"
traindate = 20150118
trainDays = 3
featurewindow = 10

itemid2category = {}
filename = "product_info_%s" % str(traindate)
for line in file(filename) :
	line = line.rstrip().split("\t")
	if len(line) < 6 : continue
	itemid, catid = line[1], line[5]
	itemid2category[itemid] = catid

targetWindows = {}
targetWindows[traindate] = 1
predictdateobj = datetime(int(str(traindate)[0:4]), int(str(traindate)[4:6]), int(str(traindate)[6:8]))
for i in range(trainDays) :
	daycnt = timedelta(days = i)
	curdate = predictdateobj - daycnt
	curdate = int(str(curdate)[:10].replace("-", ""))
	targetWindows[curdate] = 1

maxTrainDate = max(targetWindows.keys())
minTrainDate = min(targetWindows.keys())

print targetWindows, maxTrainDate, minTrainDate

fid2poscnt = {}
fid2negcnt = {}
featurename2id = {}
positiveinstance = {}
negativeinstance = {}
filename = "campany_%s_pv_clk_%s_14_days" % (IdAdvertiserCompanyId, str(traindate))
for line in file(filename) :
	line = line.rstrip().split()
	if len(line) < 5 : continue
	pyid, cvtstr, clkstr, advstr, pvstr = line[:5]

	advlist = advstr.rstrip(",").split(",")[1:]
	clklist = clkstr.rstrip(",").split(",")[1:]
	cvtlist = cvtstr.rstrip(",").split(",")[1:]

	#cvtlist = clklist

	userconvday = traindate
	label = 0
	if len(cvtlist) != 0 :
		for item in cvtlist :
			item = item.split(":")
			if len(item) < 1 : continue
			actiontime = item[0]
			if re.match("\d+$", actiontime) == None : continue
			convday = int(actiontime[:8])
			if convday in targetWindows :
				userconvday = convday
				label = 1

			if convday < minTrainDate : break

	if label != 1 :
		for item in advlist :
			item = item.split(":")
			if len(item) < 1 : continue
			actiontime, city, action = item[0], item[-1], item[-2]
			if re.match("\d+$", actiontime) == None : continue
			actionday = int(actiontime[:8])
			if actionday in targetWindows :
				userconvday = actionday
				label = 0
	
	try :
		featurestartdayobj = datetime(int(str(userconvday)[0:4]), int(str(userconvday)[4:6]), int(str(userconvday)[6:8]))
		featureenddayobj = featurestartdayobj - timedelta(featurewindow)
	except :
		continue
	featurestartday = int(str(featurestartdayobj.date()).replace("-", ""))
	featureendday = int(str(featureenddayobj.date()).replace("-", ""))

	pv = 0
	city = ""
	lastvisitday = None
	firstvisitday = None
	actionset = set()
	cateidset = set()
	action2catiddict = {}
	for item in advlist :
		item = item.split(":")
		if len(item) < 1 : continue
		actiontime, city, action = item[0], item[-1], item[-2]
		if re.match("\d+$", actiontime) == None : continue
		advday = int(actiontime[:8])
		if advday < featurestartday and advday >= featureendday :
			if lastvisitday == None : 
				lastvisitday = advday
			firstvisitday = advday
			pv += 1
			actionset.add(action)

		itemid = item[1]
		if itemid in itemid2category :
			catid = itemid2category[itemid]
			cateidset.add(catid)
	
			actioncatid = action + "_" + catid
			action2catiddict.setdefault(actioncatid, 0)
			action2catiddict[actioncatid] += 1

	if pv == 0 : continue
	if label == 0 and abs(hash(pyid) % 100) <= 70 : continue
	
	fid2fval = {}

	featurename = "constant"
	if featurename not in featurename2id :
		fid = len(featurename2id)
		featurename2id[featurename] = fid
	fid = featurename2id[featurename]
	fid2fval[fid] = 1

	featurename = "adv"
	if featurename not in featurename2id :
		fid = len(featurename2id)
		featurename2id[featurename] = fid
	fid = featurename2id[featurename]
	fval = math.log(pv + 1)
	fid2fval[fid] = fval

	try :
		predictdateobj = datetime(int(str(userconvday)[0:4]), int(str(userconvday)[4:6]), int(str(userconvday)[6:8]))
		lastvisitdayobj = datetime(int(str(lastvisitday)[0:4]), int(str(lastvisitday)[4:6]), int(str(lastvisitday)[6:8]))
		firstvisitdayobj = datetime(int(str(firstvisitday)[0:4]), int(str(firstvisitday)[4:6]), int(str(firstvisitday)[6:8]))
		lastdistance = (predictdateobj - lastvisitdayobj).days 
		firstdistance = (predictdateobj - firstvisitdayobj).days
	except :
		continue

	lastvisitdaybin = int(lastdistance)
	featurename = "lastvisitday:" + str(lastvisitdaybin)
	if featurename not in featurename2id :
		fid = len(featurename2id)
		featurename2id[featurename] = fid
	fid = featurename2id[featurename]
	fval = 1.0
	fid2fval[fid] = fval

	firstvisitdaybin = int(firstdistance)
	featurename = "firstvisitday:" + str(firstvisitdaybin)
	if featurename not in featurename2id :
		fid = len(featurename2id)
		featurename2id[featurename] = fid
	fid = featurename2id[featurename]
	fval = 1.0
	fid2fval[fid] = fval

	for actioncatid in action2catiddict :
		featurename = "actioncatid:" + actioncatid
		if featurename not in featurename2id :
			fid = len(featurename2id)
			featurename2id[featurename] = fid
		fid = featurename2id[featurename]
		fval = math.log(action2catiddict[actioncatid] + 1)
		fid2fval[fid] = fval

	for catid in cateidset :
		featurename = "catid:" + catid
		if featurename not in featurename2id :
			fid = len(featurename2id)
			featurename2id[featurename] = fid
		fid = featurename2id[featurename]
		fval = 1.0
		fid2fval[fid] = fval

	featurename = "catidpv"
	if featurename not in featurename2id :
		fid = len(featurename2id)
		featurename2id[featurename] = fid
	fid = featurename2id[featurename]
	fval = math.log(len(cateidset) * 1.0 / pv + 1)
	fid2fval[fid] = fval

	featurename = "city:" + city
	if featurename not in featurename2id :
		fid = len(featurename2id)
		featurename2id[featurename] = fid
	fid = featurename2id[featurename]
	fval = 1.0
	fid2fval[fid] = fval

	for action in actionset :
		featurename = "action:" + action
		if featurename not in featurename2id :
			fid = len(featurename2id)
			featurename2id[featurename] = fid
		fid = featurename2id[featurename]
		fval = 1.0
		fid2fval[fid] = fval

	for action in actionset :
		featurename = "action:" + action + "_city:" + city
		if featurename not in featurename2id :
			fid = len(featurename2id)
			featurename2id[featurename] = fid
		fid = featurename2id[featurename]
		fval = 1.0
		fid2fval[fid] = fval

	if label == 1 :
		for fid in fid2fval :
			fid2poscnt.setdefault(fid, 0)
			fid2poscnt[fid] += 1
		positiveinstance[pyid] = fid2fval

	if label == 0 :
		for fid in fid2fval :
			fid2negcnt.setdefault(fid, 0)
			fid2negcnt[fid] += 1
		negativeinstance[pyid] = fid2fval

# feature selection
minFeatureCnt = 5
minFeatureCnt = 10
validfidset = set()
for fid in fid2poscnt :
        if fid2poscnt[fid] >= minFeatureCnt :
                validfidset.add(fid)

postivefeaturelist = []
for pyiditem, fid2fval in positiveinstance.items() :
        featurestr = " ".join([str(x[0])+ ":" + str(x[1]) for x in sorted(fid2fval.items(), key = lambda x : x[0]) if x[0] in validfidset])
        if len(featurestr) != 0 :
                featurestr = pyiditem + "\t1\t" + featurestr
                postivefeaturelist.append(featurestr)

negativefeaturelist = []
for pyiditem, fid2fval in negativeinstance.items() :
        featurestr = " ".join([str(x[0])+ ":" + str(x[1]) for x in sorted(fid2fval.items(), key = lambda x : x[0]) if x[0] in validfidset])
        if len(featurestr) != 0 :
                featurestr = pyiditem + "\t0\t" + featurestr
                negativefeaturelist.append(featurestr)
                
print len(postivefeaturelist)
print len(negativefeaturelist)
                
globalrate = len(postivefeaturelist) * 1.0 / (len(negativefeaturelist) + len(postivefeaturelist))
featurefh = open("feature_file", "w")
for item in sorted(featurename2id.items(), key = lambda x : x[1], reverse = True) :
        fname, fid = item
        label = 0
        if fid in validfidset : label = 1
        posval = 0              
        if fid in fid2poscnt : posval = fid2poscnt[fid]
        negval = 0      
        if fid in fid2negcnt : negval = fid2negcnt[fid]
        featurefh.write(" ".join([str(fid), str(label), fname, str(posval), str(negval), str(posval * 1.0 / (posval + negval)), str(globalrate), "\n"]))
featurefh.close()

cnt = 5
trainposidx = int(len(postivefeaturelist))
trainnegidx = cnt * trainposidx
traininstancelist = postivefeaturelist
traininstancelist.extend(negativefeaturelist[:trainnegidx])
random.shuffle(traininstancelist)

trainfilename = "campaign_ext.train.%s" % (str(traindate))
trainfh = open(trainfilename, "w")
for i in range(len(traininstancelist)) :
	feature = traininstancelist[i]
        trainfh.write(feature + "\n")
trainfh.close()
