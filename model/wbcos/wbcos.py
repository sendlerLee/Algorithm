#coding:utf-8

import sys
import math
from pyspark import SparkContext, SparkConf

def user_map(line):
    cols = line.rstrip().split("\t")
    pyid = cols[0]
    proNo = cols[1]
    return pyid,{proNo:1}

def merge_dict(dictx,dicty):
    for key,val in dicty.iteritems(): 
        dictx[key] = val
    return dictx

def print_info(line):
    pyid,udict = line
    wu = 1.0 / math.log(2 + len(udict))
    return (pyid,"".join(udict.keys()),wu)

def item_map(line):
    user,itemString,wu = line
    return [(item,(wu ** 2,1)) for item in itemString.split("")]

def item_reduce(info1,info2):
    wu1,cnt1 = info1
    wu2,cnt2 = info2
    wu1 += wu2
    cnt1 += cnt2
    return (wu1,cnt1)
    
def sim_map(line):
    itemMap = itemMapBroadcast.value
    user,itemString,wu = line
    itemList = itemString.split("")
    output = []
    for item1 in itemList: 
        if item1 not in itemMap: continue
        for item2 in itemList: 
            if item1 == item2: continue
            if item2 not in itemMap: continue
            output.append(((item1,item2),(wu **2,1)))
    return output
 
def sim_calc(line):
    itemMap = itemMapBroadcast.value
    item_pair,pair_info = line  
    pair_wt,pair_cnt = pair_info 
    item1,item2 = item_pair

    item1_wt = itemMap[item1][0]
    item2_wt = itemMap[item2][0]
    sim = pair_wt / (item1_wt * item2_wt)
    return item1,{item2:sim}

def sort_sim(line):
    item,sim_dict = line
    sim_cnt = 0
    output = [] 
    for proNo,sim in sorted(sim_dict.iteritems(),key = lambda (k,v):(v,k),reverse = True):
        output.append((item,proNo,sim))
        sim_cnt += 1
        if sim_cnt >= sim_max: break
    return output

input = sys.argv[1]
output = sys.argv[2]
try:
    min_item_user = int(sys.argv[3])
    min_couser = int(sys.argv[4])
    sim_max = int(sys.argv[5]) 
except: 
    min_item_user = 2
    min_couser = 2
    sim_max = 30

sc = SparkContext()
lines = sc.textFile(input)
    
#combine user info
userInfoRdd = (lines.map(user_map)
                    .reduceByKey(merge_dict)
                    .filter(lambda x: len(x[1]) <= 200)
                    .map(print_info))
userInfoRdd.cache()
    
#cnt item info
itemInfoRdd = (userInfoRdd.flatMap(item_map)
                          .reduceByKey(item_reduce)
                          .filter(lambda x:x[1][1] >= min_item_user)
                          .map(lambda x: (x[0],(math.sqrt(x[1][0]),x[1][1]))))

itemRddMap = itemInfoRdd.collectAsMap()
itemMapBroadcast = sc.broadcast(itemRddMap)
    
#compute similarity
simRdd = (userInfoRdd.flatMap(sim_map)
                     .reduceByKey(lambda x,y:(x[0]+y[0],x[1]+y[1]))
                     .filter(lambda x: x[1][1] >= min_couser) 
                     .map(sim_calc)
                     .reduceByKey(merge_dict)
                     .flatMap(sort_sim))
#save data
simRdd.map(lambda x:"%s\t%s\t%.6f" %(x[0],x[1],x[2])).saveAsTextFile(output)  
sc.stop()
