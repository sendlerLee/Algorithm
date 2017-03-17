#!/usr/bin/python
# -*- coding:utf8 -*-

import sys
import math
import random
import re
import urlparse

if len(sys.argv) < 3 :
	print "error"
	sys.exit(-1)

inputid = sys.argv[1]
targetWin = sys.argv[2]

datelist = []
for day in targetWin.strip().split() :
	datelist.append(int(day))
datelist.sort()

targetWindows = {}
for day in datelist[:-1] :
	targetWindows.setdefault(day, 1)

maxTargetDays = max(targetWindows.keys())
print "train targetWindows = %s" % targetWindows
print "train maxTargetDays = %s" % maxTargetDays

targetUserdict = {}
featurefilename = "user_day/user_day_%s" % inputid #全部都为正样本
try:
	fd = open(featurefilename)
	for line in fd :
		line = line.rstrip().split('\t')
		if len(line) != 2 : continue
		pyid = line[0]
		advdate = line[1][:8] #有可能会出现三列，前后2行连城一块了
		if int(advdate) in targetWindows :
			targetUserdict.setdefault(pyid, 0)
except IOError:
	print "Warning: %s广告主网站没有布访客代码或访客数量太少" % featurefilename
	sys.exit(1)

convUserdict = {}
featurefilename = "conv/conv_%s" % inputid #全部都为转化样本
try:
	fd = open(featurefilename)
	for line in fd :
		line = line.rstrip().split('\t')
		pyid = line[0]
		convUserdict.setdefault(pyid, 0)
except IOError:
	print "Warning: can\'t find file or read data %s" % featurefilename

def calc_file_lines(featurelogname) :
	count = 0
	for line in file(featurelogname) :
		count += 1
	return count

pos_samples_cnt = calc_file_lines("feature_log/feature_%s_log" % inputid)

bkgcnt_thrshld = 10
if pos_samples_cnt < 6500 :
	bkgcnt_thrshld = 6
elif pos_samples_cnt < 15000 :
	bkgcnt_thrshld = 8
elif pos_samples_cnt < 35000 :
	bkgcnt_thrshld = 10
else :
	bkgcnt_thrshld = 12

url2ctr = {}
for line in file("feature_select/feature_%s_select" % inputid) :
	line = line.rstrip().split()
	if len(line) != 9 : continue
	url, ctr = line[0], line[-1]
	cnt = line[4]
	if int(cnt) <= bkgcnt_thrshld : continue
	url2ctr[url] = float(ctr)
print "len(url2ctr)=%s" % len(url2ctr)

sys.stderr.write("train 正样本个数%s... \n" % len(targetUserdict))

userid2labeldict = {}
userid2featuredict = {}
feature2labelcnt = {}
label2cnt = {}

linecnt = 0
feature2id = {}
def process(featurefilename) :
	global userid2labeldict
	global userid2featuredict
	global feature2labelcnt
	global feature2labelcnt
	global label2cnt
	global linecnt
	global feature2id
	for line in file(featurefilename) :
		line = line.rstrip().split('\t')
		if len(line) != 2 : continue
		pyid = line[0]

		label = "0"
		if pyid in targetUserdict : label = "1"
#		if label == "0" and (abs(hash(pyid)) % 1000 > 4): continue

		linecnt += 1
		'''
		if linecnt % 10000 == 0 : 
			print linecnt, label2cnt, "......"
		'''
		loopcnt = 1
		if pyid in convUserdict and pos_samples_cnt < 30000 : loopcnt = 10
		if pyid in convUserdict and pos_samples_cnt < 10000 : loopcnt = 20
		for index in xrange(0, loopcnt) :
			featureset = {}
			for urlitem in line[1].split("") :
				logtype_url = urlitem.split("")
				if len(logtype_url) < 2 : continue
				logtype = logtype_url[0]
				urlitem = logtype_url[1]
				url = urlitem.rstrip("/")
				if logtype == "bid_unbid" : 
					featureset.setdefault(logtype + "" + url, 0)
					featureset[logtype + "" + url] += 1

				urlitem = url.split("/")
				maxitem = 6
				if logtype != "bid_unbid" : maxitem = 4
				for i in range(2, min(len(urlitem), maxitem)) :
					newurl = logtype + "" + "/".join(urlitem[:i])
					featureset.setdefault(newurl, 0)
					featureset[newurl] += 1

			fidset = {}
			for url in featureset :
				if url not in url2ctr : continue

				if url not in feature2id :
					feature2id[url] = len(feature2id)
				fid = feature2id[url]

				feature2labelcnt.setdefault(fid, {})
				feature2labelcnt[fid].setdefault(label, 0)
				feature2labelcnt[fid][label] += 1

				#fidset[fid] = 1
				fidset[fid] = math.log(featureset[url] + 1.0) * url2ctr[url]

			if len(fidset) == 0 : continue
			label2cnt.setdefault(label, 0)
			label2cnt[label] += 1

			userid2labeldict[pyid] = label
			userid2featuredict[pyid] = fidset

process("feature_log/feature_0_log")
process("feature_log/feature_%s_log" % inputid)

label_pos_thrshld = 3
if pos_samples_cnt < 6000 :
	label_pos_thrshld = 2
elif pos_samples_cnt < 10000 :
	label_pos_thrshld = 3
elif pos_samples_cnt < 40000 :
	label_pos_thrshld = 4
elif pos_samples_cnt < 100000 :
	label_pos_thrshld = 5
elif pos_samples_cnt < 200000 :
	label_pos_thrshld = 6
else :
	label_pos_thrshld = 7

fcnt_thrshld = 3
if pos_samples_cnt < 5000 :
	fcnt_thrshld = 1
elif pos_samples_cnt < 15000 :
	fcnt_thrshld = 2
elif pos_samples_cnt < 25000 :
	fcnt_thrshld = 4
elif pos_samples_cnt < 50000 :
	fcnt_thrshld = 5
elif pos_samples_cnt < 100000 :
	fcnt_thrshld = 6
else :
	fcnt_thrshld = 7

# feature selection
validdict = {}
for fid in feature2labelcnt :
	if "1" not in feature2labelcnt[fid] or feature2labelcnt[fid]["1"] < label_pos_thrshld : continue
	validdict[fid] = 1

instancelist = []
neg_sample_cnt = 0
print "label2cnt[1] = %d" %(label2cnt["1"])
print "len(userid2featuredict) = %d" % (len(userid2featuredict))
print "len(validdict) = %d" % (len(validdict))
for pyid in userid2featuredict :
	label = userid2labeldict[pyid]
	fidset = userid2featuredict[pyid]
	feature = ""
	feature_count = 0
	for fid in sorted(list(fidset)) :
		if fid not in validdict : continue
		feature += str(fid) + ":" + str(fidset[fid]) + " "
		feature_count += 1
	if len(feature) == 0 : continue
	if label == "0" :
		if neg_sample_cnt > label2cnt["1"] * 1.5 : continue
		if feature_count < fcnt_thrshld : continue
		neg_sample_cnt += 1

	feature = label + " " + feature.rstrip()
	instancelist.append(feature)

featurefh = open("feature_file/feature_file_%s" % inputid, "w")
for item in sorted(feature2id.items(), key = lambda x : x[1], reverse = True) :
	catid, fid = item
	if fid not in validdict : continue
	featurefh.write(str(fid) + " " + str(catid) + "\n")
featurefh.close()

print "len(instancelist) = %d" % len(instancelist)
trainfh = open("campaign_ext/campaign_ext.train.%s" % inputid, "w")
for i in range(int(len(instancelist))) :
	curfeature = instancelist[i]
	trainfh.write(curfeature + "\n")
trainfh.close()

