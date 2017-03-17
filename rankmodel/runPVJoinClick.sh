#!/bin/sh

hdstreaming="/usr/lib/hadoop-mapreduce/hadoop-streaming.jar"

function getClkFeat()
{
	#唯品会
	IdAdvertiserCompanyId=1095

	mydate=20150605
	year=${mydate:0:4}
	month=${mydate:4:2}
	day=${mydate:6:2}

	trainsection=7

	label=train
	#label=test

	match=0
	advdatelog=/user/root/flume/express/${year}/${month}/${day}/*/adv*
	clklog=/user/root/flume/express/${year}/${month}/${day}/*/click*
	implog=/user/root/flume/express/${year}/${month}/${day}/*/imp*
	cvtlog=/user/root/flume/express/${year}/${month}/${day}/*/cvt*
        bidunbidlog=/user/root/flume/express_bid/${year}/${month}/${day}/*/*bid*
	for i in `seq 100` 
	do
		curday=`date -d "$i days ago" '+%Y%m%d'`
		if [ $curday -lt $mydate ];then
			curyear=${curday:0:4}
			curmonth=${curday:4:2}
			curday=${curday:6:2}
			newdatelog=/user/root/flume/express/${curyear}/${curmonth}/${curday}/*/adv*
			advdatelog=$advdatelog,$newdatelog

			newclklog=/user/root/flume/express/${curyear}/${curmonth}/${curday}/*/click*
			clklog=$clklog,$newclklog

                        #newdatelog=/data/production/trimmed_bid_unbid/${curyear}/${curmonth}/${curday}/*/part*
                        #bidunbidlog=$bidunbidlog,$newdatelog

			newimplog=/user/root/flume/express/${curyear}/${curmonth}/${curday}/*/imp*
			implog=$implog,$newimplog

			newcvtlog=/user/root/flume/express/${curyear}/${curmonth}/${curday}/*/cvt*
			cvtlog=$cvtlog,$newcvtlog

			match=$((match+1))
		fi

		if [ $match = $trainsection ];then
			break
		fi
	done

	echo $advdatelog
	echo $clklog
	echo $implog
	echo $bidunbidlog
	echo $cvtlog
	
	outputdir=/user/naiqiang.tan/campaign_${IdAdvertiserCompanyId}/clk/${label}/${mydate}
	echo outputdir $outputdir

	hadoop fs -rmr $outputdir
	hadoop jar ${hdstreaming} \
		-D mapreduce.job.queuename=mapreduce.important \
        	-D mapred.job.name='Campaign_Conversion' \
        	-D stream.num.map.output.key.fields=1 \
		-D stream.non.zero.exit.is.failure=false \
        	-D mapred.reduce.tasks=500 \
                -D mapred.min.split.size=1073741824 \
        	-mapper "getClickFeatMap.py" \
        	-reducer "getClickFeatReduce.py" \
        	-file getClickFeatMap.py \
        	-file getClickFeatReduce.py \
        	-file vip_itemid_catid_20150604 \
        	-input "${advdatelog},${cvtlog},${clklog}" \
        	-output "${outputdir}" 
        	#-input "${advdatelog},${cvtlog},${bidunbidlog},${clklog},${implog}" \
        	#-input "${advdatelog},${cvtlog},${clklog},${implog}" \
        	#-input "${advdatelog},${cvtlog},${clklog},${implog}" \
	hadoop fs -text ${outputdir}/part* > campany_${IdAdvertiserCompanyId}_${mydate}_${label}_feat
}
getClkFeat;
