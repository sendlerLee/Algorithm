#!/usr/bin/python
import sys

if len(sys.argv) < 3 :
    print "error parameters num"
    sys.exit(-1)

datelist = sys.argv[1]
if datelist == "" : sys.exit(-1)
train_num = int(sys.argv[2])

dat = []
arr = datelist.strip().split(' ')
for day in arr :
    dat.append(int(day))
dat.sort()

ymd = []
for dt in dat[:train_num] :
    yyyy = str(dt)[:4]
    mm = str(dt)[4:6]
    dd = str(dt)[6:8]
    ymd.append(str(yyyy)+"/"+str(mm)+"/"+str(dd))
print "NBTrain1" + "\t" + ",".join(ymd)

ymd = []
for dt in dat[train_num:] :
    yyyy = str(dt)[:4]
    mm = str(dt)[4:6]
    dd = str(dt)[6:8]
    ymd.append(str(yyyy)+"/"+str(mm)+"/"+str(dd))
print "Test1" + "\t" + ",".join(ymd)

print "NBTrain2" + "\t" + ",".join([str(dt) for dt in dat[:train_num]])
print "Test2" + "\t" + ",".join([str(dt) for dt in dat[train_num:]])

