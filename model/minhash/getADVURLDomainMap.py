#!/usr/bin/python
# -*- coding:utf8 -*-

import sys
import math
import random
import re
import urlparse
import urllib2
from urllib import unquote 

for line in sys.stdin :
	line = line.replace("\t", "")
	line = line.rstrip().split("")
	if len(line) < 3 : continue

	ActionType = line[2]
	if ActionType == "4" :
                # ADV 
                if len(line) <= 55 : continue
                ActionRequestTime, pyid, AgentReferUrl, idadvertisercompanyid, itemid = line[6], line[15], line[23], line[50], line[54]

                try :
                        res = urlparse.urlparse(AgentReferUrl)
                        netloc = res.netloc
                except :
                        netloc = "NULL"

		print netloc
