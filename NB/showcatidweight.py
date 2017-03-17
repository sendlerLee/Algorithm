#!/usr/bin/python
# -*- coding:utf8 -*-

import sys
import math
import random
import re
import urlparse

if len(sys.argv) < 2:
	print "error!\nusage:%s inputid" % __file__
	sys.exit()

#output format: url, weight, ctr,inputid
url2ctr = {}
inputid = sys.argv[1]
featurefilename = "feature_select/feature_%s_select" % inputid
for line in file(featurefilename) :
#for line in file("url_stat_ctr_two_jump_20140921") :
#for line in file("url_stat_ctr_click_20140923") :
	line = line.rstrip().split()
	url, ctr = line[0], line[-1]
	cnt = line[4]
	if int(cnt) <= 8 : continue
	url2ctr[url] = float(ctr)

fid2fname = {}
for line in file("feature_file/feature_file_%s" % inputid) :
        line = line.rstrip().split()
	if len(line) != 2 : continue
	fid, fname = line
	fid2fname[fid] = fname

fname2fval = {}
for line in file("feature_weight/feature_weight_%s" % inputid) :
        line = line.rstrip().split()
	fid, fval = line
	if fid not in fid2fname : continue
	fname = fid2fname[fid]

	fval = float(fval)
	if fname not in url2ctr : continue
	fval *= url2ctr[fname]
	#fval *= 1000
	
	fname2fval[fname] = fval

output = open("url_model_weight/url_model_weight_%s" % inputid,"w")
#sort the score=weight*ctr
for item in sorted(fname2fval.items(), key = lambda x : x[1], reverse = True) :
	fname, fval = item
	output.write("\t".join([fname, str(fval), str(url2ctr[fname]), inputid])+"\n")
