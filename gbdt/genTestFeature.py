#!/usr/bin/python
# -*- coding:utf8 -*-

import sys
import math
import random
import re
import urlparse
from datetime import *

IdAdvertiserCompanyId = "1631"
predictdate = 20150119
featurewindow = 10

itemid2category = {}
filename = "product_info_20150118"
for line in file(filename) :
	line = line.rstrip().split("\t")
	if len(line) < 6 : continue
	itemid, catid = line[1], line[5]
	itemid2category[itemid] = catid

featurename2id = {}
featurefilename = "feature_file"
for line in file(featurefilename) :
        line = line.rstrip().split()
        fid, label, feature = line[:3]
	# 有效的Feature
	if label != "1" : continue
        featurename2id[feature] = int(fid)

predictdateobj = datetime(int(str(predictdate)[0:4]), int(str(predictdate)[4:6]), int(str(predictdate)[6:8]))
oneday = timedelta(days = 1)
traindate = predictdateobj - oneday
traindate = int(str(traindate)[:10].replace("-", ""))

instancelist = []
#filename = "campany_%s_pv_clk_%s_9_days" % (IdAdvertiserCompanyId, str(traindate))
filename = "campany_%s_pv_clk_%s_14_days" % (IdAdvertiserCompanyId, str(predictdate))
for line in file(filename) :
	line = line.rstrip().split()
	if len(line) < 5 : continue
	pyid, cvtstr, clkstr, advstr, pvstr = line[:5]

	advlist = advstr.rstrip(",").split(",")[1:]
	clklist = clkstr.rstrip(",").split(",")[1:]
	cvtlist = cvtstr.rstrip(",").split(",")[1:]

	#cvtlist = clklist

	label = 0
	for item in cvtlist :
		item = item.split(":")
		if len(item) < 1 : continue
		actiontime = item[0]
		if re.match("\d+$", actiontime) == None : continue
		convday = int(actiontime[:8])
		if convday == predictdate :
			label = 1
			break

	try :
                featurestartdayobj = datetime(int(str(predictdate)[0:4]), int(str(predictdate)[4:6]), int(str(predictdate)[6:8]))
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
			cateidset.add(itemid2category[itemid])

			actioncatid = action + "_" + catid
			action2catiddict.setdefault(actioncatid, 0)
			action2catiddict[actioncatid] += 1

	if pv == 0 : continue

	fid2fval = {}

	featurename = "constant"
	if featurename in featurename2id :
		fid = featurename2id[featurename]
		fid2fval[fid] = 1

	featurename = "adv"
	if featurename in featurename2id : 
		fid = featurename2id[featurename]
		fval = math.log(pv + 1)
		fid2fval[fid] = fval

	try :
		predictdateobj = datetime(int(str(predictdate)[0:4]), int(str(predictdate)[4:6]), int(str(predictdate)[6:8]))
		lastvisitdayobj = datetime(int(str(lastvisitday)[0:4]), int(str(lastvisitday)[4:6]), int(str(lastvisitday)[6:8])) 
		firstvisitdayobj = datetime(int(str(firstvisitday)[0:4]), int(str(firstvisitday)[4:6]), int(str(firstvisitday)[6:8]))
		lastdistance = (predictdateobj - lastvisitdayobj).days
		firstdistance = (predictdateobj - firstvisitdayobj).days
	except :
		continue

	lastvisitdaybin = int(lastdistance)
	#if lastvisitdaybin > 6 : lastvisitdaybin = 6
	featurename = "lastvisitday:" + str(lastvisitdaybin)
	if featurename in featurename2id :
		fid = featurename2id[featurename]
		fval = 1.0
		fid2fval[fid] = fval

	firstvisitdaybin = int(firstdistance)
	#if firstvisitdaybin > 6 : firstvisitdaybin = 6
	featurename = "firstvisitday:" + str(firstvisitdaybin)
	if featurename in featurename2id :
		fid = featurename2id[featurename]
		fval = 1.0
		fid2fval[fid] = fval

	for catid in cateidset :
		featurename = "catid:" + catid
		if featurename in featurename2id :
			fid = featurename2id[featurename]
			fval = 1.0
			fid2fval[fid] = fval

	for actioncatid in action2catiddict :
		featurename = "actioncatid:" + actioncatid
		if featurename in featurename2id :
			fid = featurename2id[featurename]
			fval = math.log(action2catiddict[actioncatid] + 1)
			fid2fval[fid] = fval

	featurename = "catidpv"
	if featurename not in featurename2id :
		fid = len(featurename2id)
		featurename2id[featurename] = fid
	fid = featurename2id[featurename]
	fval = math.log(len(cateidset) * 1.0 / pv + 1)
	fid2fval[fid] = fval

	featurename = "city:" + city
	if featurename in featurename2id : 
		fid = featurename2id[featurename]
		fval = 1.0
		fid2fval[fid] = fval

	for action in actionset :
		featurename = "action:" + action
		if featurename in featurename2id :
			fid = featurename2id[featurename]
			fval = 1.0
			fid2fval[fid] = fval

	for action in actionset :
		featurename = "action:" + action + "_city:" + city
		if featurename in featurename2id :
			fid = len(featurename2id)
			fval = 1.0
			fid2fval[fid] = fval

        featurestr = pyid + "\t" + str(label) + "\t" + "\t".join([str(x[0])+ ":" + str(x[1]) for x in sorted(fid2fval.items(), key = lambda x : x[0])])
	instancelist.append(featurestr)

print len(instancelist)
                
testfilename = "campaign_ext.test.%s" % str(predictdate)
testfh = open(testfilename, "w")
for i in range(len(instancelist)) :
        feature = instancelist[i]
        testfh.write(feature + "\n")
testfh.close()
