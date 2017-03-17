#!/usr/bin/python
# -*- coding:utf8 -*-

import sys
import math
import random
import re
import urlparse

if len(sys.argv) < 3 :
	print "error arguments numbers"
	sys.exit(-1)

inputid = sys.argv[1]
targetWin = sys.argv[2]

datelist = []
for day in targetWin.strip().split() :
	datelist.append(int(day))
datelist.sort()

targetWindows = {}
for day in datelist[-2:] :
	targetWindows.setdefault(day, 1)
print "test targetWindows = %s" % targetWindows

maxTargetDays = max(targetWindows.keys())
targetUserdict = {}
featurefilename = "user_day/user_day_%s" % inputid
try:
	fd = open(featurefilename)
	for line in file(featurefilename) :
		line = line.rstrip().split('\t')
		if len(line) != 2 : continue
		pyid = line[0]
		advdate = line[1][:8] #有可能会出现三列，前后2行连城一块了
		if int(advdate) in targetWindows :
			targetUserdict.setdefault(pyid, 0)
except IOError:
	print "Warning: %s广告主网站没有布访客代码或访客数量太少" % featurefilename
	sys.exit(1)

def calc_file_lines(featurelogname) :
	count = 0
	for line in file(featurelogname) :
		count += 1
	return count

pos_samples_cnt = calc_file_lines("feature_log/feature_%s_log" % inputid)

url2fid = {}
featurefilename = "feature_file/feature_file_%s" % inputid
for line in file(featurefilename) :
	line = line.rstrip().split()
	fid, url = line
	url2fid[url] = fid

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

sys.stderr.write("test 正样本个数%s... \n" % len(targetUserdict))

userid2labeldict = {}
userid2featuredict = {}
label2cnt = {}

MinFreqThreshold = 1

linecnt = 0
def process(featurefilename) :
	global userid2labeldict
	global userid2featuredict
	global label2cnt
	global linecnt
	for line in file(featurefilename) :
		line = line.rstrip().split('\t')
		if len(line) != 2 : continue
		pyid = line[0]

		label = "0"
		if pyid in targetUserdict : label = "1"
#		if label == "0" and (abs(hash(pyid)) % 1000 < 4 or abs(hash(pyid)) % 1000 >= 10): continue

		linecnt += 1
		'''
		if linecnt % 10000 == 0 :
			print linecnt, label2cnt, "......"
		'''

		fidset = {}
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
			if url not in url2fid : continue
			fid = url2fid[url]
			fidset[fid] = math.log(featureset[url] + 1.0) * url2ctr[url]

		if len(fidset) == 0 : continue
		label2cnt.setdefault(label, 0)
		label2cnt[label] += 1

		userid2labeldict[pyid] = label
		userid2featuredict[pyid] = fidset

process("feature_log/feature_0_log")
process("feature_log/feature_%s_log" % inputid)

instancelist = []
for pyid in userid2featuredict :
	label = userid2labeldict[pyid]
	fidset = userid2featuredict[pyid]
	feature = ""
	for fid in sorted(list(fidset)) :
		feature += str(fid) + ":" + str(fidset[fid]) + " "
		#feature += str(fid) + ":" + str(1) + " "
	if len(feature) == 0 : continue

	feature = label + " " + feature.rstrip()
	instancelist.append(feature)

testfh = open("campaign_ext/campaign_ext.test.%s" % inputid, "w")
for i in range(int(len(instancelist))) :
	curfeature = instancelist[i]
	testfh.write(curfeature + "\n")
testfh.close()

