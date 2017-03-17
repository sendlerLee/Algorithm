#!/usr/bin/python
# -*- coding:utf8 -*-

import sys
import math
import random
import re
import urlparse
import urllib2
from urllib import unquote 

filename = "url_cooc_20150324"

url2cnt = {}
for line in file(filename) :
	line = line.rstrip().split()
	if len(line) != 2 : continue
	urlcooc, cnt = line
	cnt = int(cnt)

	urlcooc = urlcooc.split("")
	if len(urlcooc) == 1 :
		url = urlcooc[0]
		url2cnt[url] = cnt

for line in file(filename) :
	line = line.rstrip().split()
	if len(line) != 2 : continue
	urlcooc, cnt = line

	urlcooc = urlcooc.split("")
	if len(urlcooc) == 2 :
		url, other_url = urlcooc
		if other_url not in url2cnt : continue
		print url, other_url, cnt, url2cnt[other_url], int(cnt) * 100.0 / url2cnt[other_url]
