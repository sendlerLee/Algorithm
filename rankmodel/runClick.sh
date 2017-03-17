#!/bin/sh


CompanyId=1095
traindate=20150604

train=campany_${CompanyId}_${traindate}_train_feat

traindate=20150605
predict=campany_${CompanyId}_${traindate}_test_feat

result=campaign_clk.plr.result.${traindate}

echo $train  $predict

#../logitboost/PLR -i 80 -s 1 -l 0.001 -r 0.0005 $train $predict $result
#../logitboost/PLR -i 80 -l 0.0001 -r 0.0001 -s 2 $train $predict $result

#../logitboost/PLR_L1 -i 80 -l 0.0001 -r 0.0001 -s 5 $train $predict $result
#../logitboost/LTOR -i 80 -l 0.001 -r 0.001 -s 5 $train $predict $result
../logitboost/LTOR -i 10 -l 0.01 -r 0.01 -s 5 $train $predict $result

#../logitboost/SSFALR -i 10 -l 0.01 $train $predict $result

#../randomforest/randomforest -d 5 -m 10 -s 20 -t 50 $train $predict $result

#
#../lr_gdbt/gbdt -d 5 -m 10 -s 20 -l 0.01 -t 50 $train $predict $result
#../gbdt/gbdt -d 5 -m 10 -s 20 -l 0.01 -t 50 $train $predict $result
#../poisson_boosting/gbdt -d 5 -m 10 -s 20 -l 0.001 -t 100 $train $predict $result

#../adaboost2/gbdt -d 5 -m 10 -s 20 -l 0.01 -t 50 $train $predict $result
#../adaboost_book/gbdt -d 5 -m 10 -s 20 -l 0.01 -t 50 $train $predict $result
#../adaboost_my/gbdt -d 5 -m 10 -s 20 -l 0.01 -t 20 $train $predict $result
