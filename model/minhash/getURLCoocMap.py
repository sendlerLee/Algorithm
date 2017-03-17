#!/usr/bin/python
# -*- coding:utf8 -*-

import sys
import math
import random
import re
import urlparse
import urllib2
from urllib import unquote 

url2pv = {}
for line in file("url_pv_sort") :
	line = line.rstrip().split()
	if len(line) != 2 : continue
	url, pv = line
	url2pv[url] = int(pv)
	if len(url2pv) >= 5000 : break

for line in file("adv_netloc_cnt") :
	line = line.rstrip().split()
	if len(line) != 2 : continue
	url, pv = line
	if int(pv) < 500 : continue
	url2pv[url] = int(pv)

for line in sys.stdin :
	line = line.rstrip().split("")
	if len(line) <= 2 : continue

	urlset = set()
	for item in line[1:] :
		item = item.split("")
		if len(item) != 2 : continue
		url, cnt = item
		if url not in url2pv : continue
		urlset.add(url)

	urllist = sorted(list(urlset))
	for i in range(len(urllist)) :
		print urllist[i]

	if len(urllist) >= 2 :
		for i in range(len(urllist) - 1) :
			for j in range(i + 1, len(urllist)) :
				print urllist[i] + "" + urllist[j]
