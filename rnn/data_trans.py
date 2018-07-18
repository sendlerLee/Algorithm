#!/usr/bin/env python

import sys

link_dict={}
for line in sys.stdin :
    line = line.strip().split(" ")
    for link in line[2:] :
        link = link.replace(":1","")
        link_dict.setdefault(link,0)
        link_dict[link] += 1 

for link,num in sorted(link_dict.items(),key=lambda x:x[1],reverse=True) :
    print (link,num)
