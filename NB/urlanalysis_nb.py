#!/usr/bin/python
# -*- coding:utf8 -*-

import sys
import math
import random
import re
import urlparse

dirpath = sys.argv[1]
inputid = sys.argv[2]

cheat_domain = {}
for line in file("cheat_domain.txt") :
	line = line.rstrip().split('\t')
	if len(line) < 1 : continue
	cheat_domain.setdefault(line[0], 1)
shield_domain = {}
for line in file("shield_domain.txt") :
	line = line.rstrip().split('\t')
	if len(line) < 1 : continue
	shield_domain.setdefault(line[0], 1)

poscnt = 0
negcnt = 0
feature2poscnt = {}
feature2negcnt = {}
feature2bkgcnt = {}

#positive examples
featurefilename = "%s/feature_%s_log" % (dirpath, inputid)
for line in file(featurefilename) :
	line = line.rstrip().split('\t')
	if len(line) < 2 : continue
	pyid = line[0]

	featureset = set()
	for url in line[1].split("") :
		logtype_url = url.split("")
		if len(logtype_url) < 2 : continue
		logtype = logtype_url[0]
		url = logtype_url[1]
		if url.find("http://news.")>=0 or url.find("http://ent.")>=0 or url.find("http://photo.")>=0: continue
		if url.find("http%3A")>=0 : continue
		if url.startswith("file:") : continue
		domain = urlparse.urlparse(url)[1]
		if domain in cheat_domain : continue
		if domain in shield_domain : continue
		url = url.rstrip("/")
		if logtype == "bid_unbid" : featureset.add(logtype+""+url)
		urlitem = url.split("/")
		maxitem = 6
		if logtype != "bid_unbid" : maxitem = 4
		for i in range(3, min(len(urlitem), maxitem + 1)) :
			newurl = logtype + "" + "/".join(urlitem[:i])
			featureset.add(newurl)

	poscnt += 1
	for feature in featureset :
		feature2poscnt.setdefault(feature, 0)
		feature2poscnt[feature] += 1
		feature2bkgcnt.setdefault(feature, 0)
		feature2bkgcnt[feature] += 1

if poscnt == 0 : sys.exit(-1)
#print feature2poscnt

#negative examples
featurefilename = "%s/feature_0_log" % (dirpath)
for line in file(featurefilename) :
	line = line.rstrip().split('\t')
	if len(line) < 2 : continue
	pyid = line[0]

	featureset = set()
	for url in line[1].split("") :
		logtype_url = url.split("")
		if len(logtype_url) < 2 : continue
		logtype = logtype_url[0]
		url = logtype_url[1]
		if url.find("http://news.")>=0 or url.find("http://ent.")>=0 or url.find("http://photo.")>=0: continue
		if url.find("http%3A")>=0 : continue
		if url.startswith("file:") : continue
		domain = urlparse.urlparse(url)[1]
		if domain in cheat_domain : continue
		if domain in shield_domain : continue
		url = url.rstrip("/")
		if logtype == "bid_unbid" : featureset.add(logtype+""+url)
		urlitem = url.split("/")
		maxitem = 6
		if logtype != "bid_unbid" : maxitem = 4
		for i in range(3, min(len(urlitem), maxitem + 1)) :
			newurl = logtype + "" + "/".join(urlitem[:i])
			featureset.add(newurl)
	negcnt += 1

	for feature in featureset :
		feature2negcnt.setdefault(feature, 0)
		feature2negcnt[feature] += 1
		feature2bkgcnt.setdefault(feature, 0)
		feature2bkgcnt[feature] += 1

ctr_thrshld = 0.2
lift_thrshld = 0.7
bkgcnt_thrshld = 3
if poscnt > 6500 and poscnt < 10000 :
	bkgcnt_thrshld = 4
elif poscnt >= 10000:
	bkgcnt_thrshld = 5

bkgcnt = poscnt + negcnt
for feature, cnt in sorted(feature2poscnt.items(), key = lambda x : x[1], reverse = True) :
	if feature not in feature2negcnt : continue

	ctr = (feature2poscnt[feature] + 1) * 1.0 / (feature2negcnt[feature] + feature2poscnt[feature] + 5)

	posrate = feature2poscnt[feature] * 1.0 / poscnt
	bkgrate = feature2bkgcnt[feature] * 1.0 / bkgcnt
	lift = posrate / (posrate + bkgrate)
	if feature2poscnt[feature] >= 1 :
#		if poscnt > 10000 : lift_thrshld = 0.85
		if feature not in feature2bkgcnt or lift < lift_thrshld : continue
		if poscnt > 5000 :
			if feature2poscnt[feature] <= 1: continue
		if poscnt > 1800 :
			if feature2negcnt[feature] <= 1: continue
		if feature2bkgcnt[feature] < bkgcnt_thrshld and feature2poscnt[feature] <= 2: continue
#		if ctr < ctr_thrshld : continue
		print feature, feature2poscnt[feature], poscnt, posrate, feature2bkgcnt[feature], bkgcnt, bkgrate, lift, ctr
