#!/usr/bin/python
# -*- coding:utf8 -*-

import sys
import math
import random
import re
import urlparse
import urllib2

COMPANYID = "1498"

COMPANYID = "1123"

COMPANYID = "1741"

COMPANYID = "1052"

COMPANYID = "1631"

#COMPANYID = "1352"

#COMPANYID = "955"

COMPANYID = "1123"

#COMPANYID = "3044"

COMPANYID = "1095"

#COMPANYID = "2878"

for line in sys.stdin :
	line = line.replace("\t", "")
	line = line.rstrip().split("")
	if len(line) <= 2 : continue

	ActionType = line[2]

	if ActionType == "8" :
		# bid 
		if len(line) <= 90 : continue
		ActionRequestTime, user, idadvertisercompanyid, IdProductNo = line[6], line[15], line[64], line[58]
		if len(user) == 0 : continue
		AgentUrl = line[21]
		PayBidPrice = line[90]
		if ActionRequestTime.isdigit() :
			print "\t".join([user, "bidunbid", ActionRequestTime + "" + AgentUrl + "" + PayBidPrice + "" + idadvertisercompanyid])

	if ActionType == "3" :
                # Conversion 
                if len(line) <= 68 : continue
                ActionRequestTime, user, idadvertisercompanyid, ProductList = line[6], line[15], line[50], line[66]
                if idadvertisercompanyid != COMPANYID : continue
		TargetType = line[51]

		if ActionRequestTime.isdigit() :
			ProductList = ";".join(ProductList.split())
                	print "\t".join([user, "cvt", ActionRequestTime + "" + TargetType + "" + ProductList])

        if ActionType == "1" :
                # Impression 
                if len(line) <= 59 : continue
                ActionRequestTime, user, idadvertisercompanyid, IdProductNo = line[6], line[15], line[64], line[58]
                IdStrategyId = line[53]
                if idadvertisercompanyid != COMPANYID : continue
		AgentUrl = line[21]
                if ActionRequestTime.isdigit() :
                        #print "\t".join([user, "pv", ActionRequestTime + "" + IdStrategyId + "" + IdProductNo + "" + AgentUrl])
                        print "\t".join([user, "pv", ActionRequestTime + "" + AgentUrl])

        if ActionType == "2" :
                # Click  
                if len(line) <= 84 : continue
                ActionRequestTime, user, idadvertisercompanyid, IdProductNo = line[6], line[15], line[64], line[58]
                if idadvertisercompanyid != COMPANYID : continue
		AgentUrl = line[21]
		IdStrategyId = line[53]
                if ActionRequestTime.isdigit() :
                        #print "\t".join([user, "clk", ActionRequestTime + "" + IdStrategyId + "" + IdProductNo + "" + AgentUrl])
                        print "\t".join([user, "clk", ActionRequestTime + "" + IdStrategyId + "" + IdProductNo])

	if ActionType == "4" :
		# ADV 
		if len(line) <= 55 : continue
        	ActionRequestTime, user, AgentReferUrl, idadvertisercompanyid, itemid = line[6], line[15], line[23], line[50], line[54]
		AgentUrl = line[21]
		GeoId = line[34]
        	if idadvertisercompanyid != COMPANYID : continue

                ActionUri = line[20]
		utm_medium = ""
                utm_source = ""
                utm_campaign = ""
                try :
                        #newurl = urllib2.unquote(ActionUri).decode('utf-8')
                        newurl = urllib2.unquote(AgentUrl).decode('utf-8')
			
                        for url in newurl.split("?") :
                                for item in  url.split("&") :
                                        item = item.split("=")
                                        if len(item) != 2 : continue
                                        if item[0] == "utm_campaign" :
                                                utm_campaign = item[1]
                                        if item[0] == "utm_source" :
                                                utm_source = item[1]
                                        if item[0] == "utm_medium" :
                                                utm_medium = item[1]
                        utm_source = str(utm_source) + "" + str(utm_campaign)
			if re.match("\d+$", ActionRequestTime) != None :
				print "\t".join([user, "adv", ActionRequestTime + "" + itemid + "" + AgentUrl + "" + utm_medium + "" + GeoId])
                except :
                        continue
