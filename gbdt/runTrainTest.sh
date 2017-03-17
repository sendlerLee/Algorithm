#!/bin/sh

#./gbdt -d 7 -s 20 -t 10 campaign_ext.train.dangdang campaign_ext.test.dangdang campaign_ext.train.result campaign_ext.test.result 

#./gbdt -d 7 -s 20 -t 10 campaign_ext.train campaign_ext.test campaign_ext.train.result campaign_ext.test.result 
#./gbdt -d 7 -s 20 -t 50 campaign_ext.train campaign_ext.test campaign_ext.train.result campaign_ext.test.result 

#dangdang
./gbdt -d 7 -m 350 -s 20 -t 10 campaign_ext.train.dangdang.all campaign_ext.test.dangdang.all campaign_ext.train.result campaign_ext.test.result 
#./gbdt -d 7 -m 350 -s 20 -t 50 campaign_ext.train.dangdang.subcat campaign_ext.test.dangdang.subcat campaign_ext.train.result campaign_ext.test.result 
#./gbdt -d 3 -m 500 -s 20 -t 5 campaign_ext.train.dangdang campaign_ext.test.dangdang campaign_ext.train.result2 campaign_ext.test.result2

#./gbdt -d 7 -s 20 -t 50 campaign_ext.train.without.subcat campaign_ext.test.without.subcat campaign_ext.train.result campaign_ext.test.result 
#./gbdt -d 7 -m 50 -s 20 -t 50 campaign_ext.train.without.subcat campaign_ext.test.without.subcat campaign_ext.train.result campaign_ext.test.result 

# Secoo
#./gbdt -d 4 -m 350 -s 20 -t 10 campaign_ext.train.secoo.20150113 campaign_ext.test.secoo.20150113 campaign_ext.train.result campaign_ext.test.result 

#./gbdt -d 5 -s 20 -t 100 campaign_ext.train.lifevc campaign_ext.test.lifevc campaign_ext.train.result campaign_ext.test.result 
#./gbdt -d 5 -s 20 -t 130 campaign_ext.train.lifevc2 campaign_ext.test.lifevc2 campaign_ext.train.result campaign_ext.test.result 

#lifevc
#./gbdt -d 7 -m 350 -s 20 -t 20 campaign_ext.train.lifevc.20150114 campaign_ext.test.lifevc.20150115 campaign_ext.train.result campaign_ext.test.result 
#./gbdt -d 5 -m 500 -s 20 -t 1 campaign_ext.train.lifevc.20150114.trim campaign_ext.test.lifevc.20150115.trim campaign_ext.train.result campaign_ext.test.result 