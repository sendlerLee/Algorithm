#!/usr/bin/python
# -*- coding:utf8 -*-

import sys
import math
import random
import re
import datetime 
import time

traindate = "20150604"
trainDays = 3

#traindate = "20150605"
#trainDays = 1

featurewindow = 7

itemid2category = {}
filename = "vip_itemid_catid_20150604"
for line in file(filename) :
        line = line.rstrip().split()
        if len(line) < 2 : continue
        itemid, catid = line[0], line[1:]
	catid = "".join(catid)
        itemid2category[itemid] = catid

targetWindows = {}
targetWindows[traindate] = 1
predictdateobj = datetime.datetime(int(traindate[0:4]), int(traindate[4:6]), int(traindate[6:8]))
for i in range(trainDays) :
        daycnt = datetime.timedelta(days = i)
        curdate = predictdateobj - daycnt
        curdate = str(curdate)[:10].replace("-", "")
        targetWindows[curdate] = 1

def getClickItem(datetime2clkitems) :
	clkitem = None
	clktime = None
	for item in sorted(datetime2clkitems.items(), key = lambda x : x[0], reverse = True) :
			item = item[1]
			item = item.split("")
			if len(item) < 3 : continue
			actiontime, IdStrategyId, IdProductNo = item[:3]
                        convday = actiontime[:8]
                        if convday in targetWindows and len(IdProductNo) != 0 :
                                clktime = actiontime
				clkitem = IdProductNo
                                break

	if clkitem != None :
		clkitem = clkitem.strip("+")
	return clkitem, clktime
		

def getClickFeat(pyid, clkitem, clktime, datetime2advitems) :
	featureenddayobj = datetime.datetime(int(clktime[0:4]), int(clktime[4:6]), int(clktime[6:8]), int(clktime[8:10]), int(clktime[10:12]), int(clktime[12:14]))
	featurestartdayobj = featureenddayobj - datetime.timedelta(featurewindow)

	itemid2seq = {}
	lastclkitemid = None
	itemidset = set()
	userfeatdict = {}
	for item in sorted(datetime2advitems.items(), key = lambda x : x[0], reverse = True) :
		item = item[1]
		item = item.split("")
		if len(item) != 5 : continue
		actiontime, itemid, agenturl, utm_medium, geoid = item
		actiontimeobj = datetime.datetime(int(actiontime[0:4]), int(actiontime[4:6]), int(actiontime[6:8]), int(actiontime[8:10]), int(actiontime[10:12]))
		if actiontimeobj >= featurestartdayobj and actiontimeobj < featureenddayobj : 
			if len(itemid) != 0 :
				userfeatdict.setdefault(itemid, 0)
				userfeatdict[itemid] += 1

				itemidset.add(itemid)

				if itemid in itemid2category :
					catid = itemid2category[itemid]
					userfeatdict.setdefault(catid, 0)
					userfeatdict[catid] += 1

					if len(utm_medium) != 0 :
						action = "dspclk_" + catid
						userfeatdict.setdefault(action, 0)
						userfeatdict[action] += 1

				if len(utm_medium) != 0 :
					action = "dspclk_" + itemid
					userfeatdict.setdefault(action, 0)
					userfeatdict[action] += 1

				if lastclkitemid == None :
					lastclkitemid = itemid

				itemid2seq[itemid] = len(itemid2seq)

			userfeatdict[geoid] = 1

	if len(userfeatdict) == 0 : return None, None

	# 没有浏览过的商品
	if clkitem not in itemid2seq : return None, None

	clkitemfeatdict = {}
	clkitemfeatdict[clkitem] = 1
	if clkitem in itemid2category :
		catid = itemid2category[clkitem]
		clkitemfeatdict[catid] = 1

	userclkfeatdict = {}
	for uft in userfeatdict :
		for ift in clkitemfeatdict :
			feat = uft + "_" + ift
			userclkfeatdict[feat] = 1
			
	userfeat = "\t".join(["u_"  + feat + ":1" for feat in userfeatdict])
	itemfeat = "\t".join(["i_"  + feat + ":1" for feat in clkitemfeatdict])
	useritemfeat = "\t".join(["ui_"  + feat + ":1" for feat in userclkfeatdict])

	ctxfeat = ""
	if clkitem == lastclkitemid :
		ctxfeat += "c_l:1"

	if clkitem in itemid2seq :
		seq = itemid2seq[clkitem]
		feat = "c_s_" + str(seq)
		ctxfeat += "\t" + "\t".join([feat + ":1"])

	clkfeat = pyid + ":" + clkitem + "\t1\t" + "\t".join([userfeat, itemfeat, useritemfeat, ctxfeat])

	SampleCnt = 6

	nonclkfeatlist = []
	candidateitemidlist = [itemid for itemid in itemidset if itemid != clkitem][:SampleCnt]
	for nonclkitem in candidateitemidlist :
		#nonclkitem = random.choice(candidateitemidlist)

		nonclkitemfeatdict = {}
		nonclkitemfeatdict[nonclkitem] = 1
		if nonclkitem in itemid2category :
			catid = itemid2category[nonclkitem]
			nonclkitemfeatdict[catid] = 1

		usernonclkfeatdict = {}
		for uft in userfeatdict :
			for ift in nonclkitemfeatdict :
				feat = uft + "_" + ift
				usernonclkfeatdict[feat] = 1
			
		nonclkitemfeat = "\t".join(["i_"  + feat + ":1" for feat in nonclkitemfeatdict])
		usernonitemfeat = "\t".join(["ui_"  + feat + ":1" for feat in usernonclkfeatdict])

		ctxfeat = ""
		if nonclkitem == lastclkitemid :
			ctxfeat += "c_l:1"

		if nonclkitem in itemid2seq :
			seq = itemid2seq[nonclkitem]
			feat = "c_s_" + str(seq)
			ctxfeat += "\t" + "\t".join([feat + ":1"])

		nonclkfeat = pyid + ":" + nonclkitem + "\t0\t" + "\t".join([userfeat, nonclkitemfeat, usernonitemfeat, ctxfeat])
		nonclkfeatlist.append(nonclkfeat)

	if len(nonclkfeatlist) == 0 : return None, None
	return clkfeat, nonclkfeatlist

lastuser = None
datetime2advitems = {}
datetime2cvtitems = {}
datetime2bidunbid = {}
datetime2clkitems = {}
datetime2pvitems = {}
for line in sys.stdin :
	line = line.rstrip().split("\t")
	if len(line) != 3 : continue
	user, ActionType, IdProductNo = line

	if user != lastuser and lastuser != None :
		if len(datetime2clkitems) != 0 and len(datetime2advitems) <= 10000 :
			clkitem, clktime = getClickItem(datetime2clkitems)
			if clkitem != None : 
				clkfeat, nonclkfeatlist = getClickFeat(lastuser, clkitem, clktime, datetime2advitems)
				if clkfeat != None or nonclkfeatlist != None : 
					print clkfeat
					for nonclkfeat in nonclkfeatlist :
						print nonclkfeat
			
		datetime2cvtitems = {}
		datetime2advitems = {}
		datetime2bidunbid = {}
		datetime2clkitems = {}
		datetime2pvitems = {}

	if ActionType == "clk" :
		if len(datetime2clkitems) <= 100 : 
                	IdProductNolist = IdProductNo.split("")
                	if len(IdProductNolist) > 0 : 
                		ActionRequestTime = IdProductNolist[0]
                		datetime2clkitems[ActionRequestTime] = IdProductNo

	if ActionType == "pv" :
		if len(datetime2pvitems) <= 100 : 
                	IdProductNolist = IdProductNo.split("")
                	if len(IdProductNolist) > 0 : 
                		ActionRequestTime = IdProductNolist[0]
                		datetime2pvitems[ActionRequestTime] = IdProductNo

	if ActionType == "cvt" :
		if len(datetime2cvtitems) <= 100 : 
                	IdProductNolist = IdProductNo.split("")
                	if len(IdProductNolist) > 0 : 
                		ActionRequestTime = IdProductNolist[0]
                		datetime2cvtitems[ActionRequestTime] = IdProductNo

	if ActionType == "bidunbid" :
		if len(datetime2bidunbid) <= 3000 : 
			IdProductNolist = IdProductNo.split("")
			if len(IdProductNolist) > 0 : 
				ActionRequestTime = IdProductNolist[0]
				datetime2bidunbid[ActionRequestTime] = IdProductNo

	if ActionType == "adv" :
		if len(datetime2advitems) <= 1000 : 
			IdProductNolist = IdProductNo.split("")
			if len(IdProductNolist) > 0 : 
				ActionRequestTime = IdProductNolist[0]
				datetime2advitems[ActionRequestTime] = IdProductNo

	lastuser = user
