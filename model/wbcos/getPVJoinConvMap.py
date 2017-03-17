#!/usr/bin/python
# -*- coding:utf8 -*-

import sys
import math
import random
import re
import urlparse
import urllib2

for line in sys.stdin :
	line = line.replace("\t", "")
	line = line.rstrip().split("")
	if len(line) <= 2 : continue

	ActionType = line[2]

	if ActionType == "4" or ActionType == "8" :
		# ADV 
		if len(line) <= 42 : continue

		pyid = line[15]
		AgentUrl = "".join(line[21].split())
		DeviceType = line[41]
		if DeviceType != "General" : continue

		domain = None
                try :
                        res = urlparse.urlparse(AgentUrl)
                        domain = res.netloc
                        domain = "".join(domain.split())
                except :
                        continue

		if len(pyid) > 0 and len(domain) <= 100 and len(domain) > 0 :
			#print "\t".join([pyid, AgentUrl])
			print "\t".join([pyid, domain])

	"""
	if ActionType == "8" :
		# BIDUNBID 
		if len(line) <= 22 : continue

		pyid = line[15]
		AgentUrl = "".join(line[21].split())

		print "\t".join([pyid, AgentUrl])
	"""
