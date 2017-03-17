#!/bin/sh

hdstreaming="/usr/lib/hadoop-mapreduce/hadoop-streaming.jar"

function genInstance()
{
	mydate=20150818
        year=${mydate:0:4}
	month=${mydate:4:2}
	day=${mydate:6:2}

	trainsection=1

	match=0
	advdatelog=/user/root/flume/express/${year}/${month}/${day}/*/adv*
	bidunbidlog=/user/root/flume/express_bid/${year}/${month}/${day}/*/*bid*
	for i in `seq 100` 
	do
		curday=`date -d "$i days ago" '+%Y%m%d'`
		if [ $curday -lt $mydate ];then
			curmonth=${curday:4:2}
			curday=${curday:6:2}
			newdatelog=/user/root/flume/express/${year}/${curmonth}/${curday}/*/adv*
			advdatelog=$advdatelog,$newdatelog

			newdatelog=/user/root/flume/express_bid/${year}/${month}/${day}/*/*bid*
                	bidunbidlog=$bidunbidlog,$newdatelog

			match=$((match+1))
		fi

		if [ $match = $trainsection ];then
			break
		fi
	done

	advdatelog=/user/root/flume/express/${year}/${month}/${day}/*/adv*
	bidunbidlog=/user/root/flume/express_bid/${year}/${month}/${day}/*/*bid*

	echo $advdatelog
	echo $bidunbidlog

	# 要在Map中指定文件名
	outputdir=/user/naiqiang.tan/url_similarity/${mydate}
	hadoop fs -rmr $outputdir
	hadoop jar ${hdstreaming} \
		-D mapreduce.job.queuename=mapreduce.important \
        	-D mapred.job.name='UserFeature::URL' \
        	-D stream.num.map.output.key.fields=1 \
        	-D mapred.reduce.tasks=50 \
        	-mapper "getPVJoinConvMap.py" \
        	-reducer "getPVJoinConvReduce.py" \
        	-file getPVJoinConvMap.py \
        	-file getPVJoinConvReduce.py \
        	-input "${advdatelog},${bidunbidlog}" \
        	-output "${outputdir}"

        	#-input "${advdatelog}" \
	hadoop fs -text ${outputdir}/part* > url_similarity_${mydate}_${trainsection}_train_instance
} 
genInstance;
