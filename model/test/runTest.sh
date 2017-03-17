#!/bin/sh


train=traindata
test=testdata

#train=traindata.gbdtfeat
#test=testdata.gbdtfeat

featweight=test_feat
/data/users/naiqiang.tan/model/linearmodel/MyBoost -i 2000 -l 0.1 $train $test $featweight

featweight=test_feat_plb
#/data/users/naiqiang.tan/model/linearmodel/PLogitBoost -i 2000 -l 0.01 $train $test $featweight

#/data/users/naiqiang.tan/model/linearmodel/swaf -i 200 -l 0.01 $train $test $featweight
#/data/users/naiqiang.tan/model/linearmodel/FTRLProximal  $train $test $featweight
#/data/users/naiqiang.tan/model/gbdt/gbdt -t 50 -d 8 -l 0.02 -f 1 $train $test $featweight
#/data/users/naiqiang.tan/model/randomforest/randomforest -t 100 -d 4 $train $test $featweight

featweight=lr_test_feat
#/data/users/naiqiang.tan/model/linearmodel/BinLR -i 50 -l 0.002 -r 0.001 $train $test $featweight
