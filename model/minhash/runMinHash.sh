#!/bin/sh

hdstreaming="/usr/lib/hadoop-mapreduce/hadoop-streaming.jar"

function genInstance()
{
	IdAdvertiserCompanyId=2448
	
	# 房点通
	#IdAdvertiserCompanyId=2188
	mydate=20150305
	month=${mydate:4:2}
	day=${mydate:6:2}

	match=0
	trainsection=6
	bidunbidlog=/data/production/url_set_pyid/2015/${month}/${day}/*/part*
	for i in `seq 100` 
	do
		curday=`date -d "$i days ago" '+%Y%m%d'`
		if [ $curday -lt $mydate ];then
			curmonth=${curday:4:2}
			curday=${curday:6:2}

			newdatelog=/data/production/url_set_pyid/2015/${curmonth}/${curday}/*/part*
                	bidunbidlog=$bidunbidlog,$newdatelog

			match=$((match+1))
		fi

		if [ $match = $trainsection ];then
			break
		fi
	done

	bidunbidlog=/data/production/url_set_pyid/2015/${month}/${day}/*/part*

	echo $bidunbidlog

	# 要在Map中指定文件名
	outputdir=/user/naiqiang.tan/minhash/urlhashval/${mydate}
	hadoop fs -rmr $outputdir
	hadoop jar ${hdstreaming} \
		-D mapreduce.job.queuename=mapreduce.important \
        	-D mapred.job.name='UserFeature::MinHash' \
        	-D stream.num.map.output.key.fields=1 \
        	-D mapred.reduce.tasks=50 \
        	-mapper "getURLMinHashMap.py" \
        	-reducer "getURLMinHashReduce.py" \
        	-file getURLMinHashMap.py \
        	-file getURLMinHashReduce.py \
        	-input "${bidunbidlog}" \
        	-output "${outputdir}"
	hadoop fs -text ${outputdir}/part* > minhash_${mydate}_${trainsection}
} 
#genInstance;

function mergeUserURL()
{
	IdAdvertiserCompanyId=2448
	
	# 房点通
	#IdAdvertiserCompanyId=2188
	mydate=20150307
	year=${mydate:0:4}
	month=${mydate:4:2}
	day=${mydate:6:2}

	match=0
	trainsection=6
	bidunbidlog=/data/production/url_set_pyid/${year}/${month}/${day}/*/part*
	for i in `seq 100` 
	do
		curday=`date -d "$i days ago" '+%Y%m%d'`
		if [ $curday -lt $mydate ];then
			curyear=${mydate:0:4}
			curmonth=${curday:4:2}
			curday=${curday:6:2}

			newdatelog=/data/production/url_set_pyid/${curyear}/${curmonth}/${curday}/*/part*
                	bidunbidlog=$bidunbidlog,$newdatelog

			match=$((match+1))
		fi

		if [ $match = $trainsection ];then
			break
		fi
	done

	#bidunbidlog=/data/production/url_set_pyid/${year}/${month}/${day}/*/part*

	echo $bidunbidlog

	# 要在Map中指定文件名
	outputdir=/user/naiqiang.tan/minhash/megeruserurl/${mydate}
	hadoop fs -rmr $outputdir
	hadoop jar ${hdstreaming} \
		-D mapreduce.job.queuename=mapreduce.important \
        	-D mapred.job.name='UserFeature::MergeUserURL' \
        	-D stream.num.map.output.key.fields=1 \
        	-D mapred.reduce.tasks=50 \
        	-mapper "mergeUserURLMap.py" \
        	-reducer "mergeUserURLReduce.py" \
        	-file mergeUserURLMap.py \
        	-file mergeUserURLReduce.py \
        	-input "${bidunbidlog}" \
        	-output "${outputdir}"
} 
mergeUserURL;

function getURLCooc()
{
	mydate=20150307
	inputdir=/user/naiqiang.tan/minhash/megeruserurl/${mydate}

	echo $inputdir

	# 要在Map中指定文件名
	outputdir=/user/naiqiang.tan/minhash/urlcooc/${mydate}
	hadoop fs -rmr $outputdir

	hadoop jar ${hdstreaming} \
		-D mapreduce.job.queuename=mapreduce.important \
        	-D mapred.job.name='UserFeature::URLCooc' \
        	-D stream.num.map.output.key.fields=1 \
        	-D mapred.reduce.tasks=50 \
        	-mapper "getURLCoocMap.py" \
        	-reducer "getURLCoocReduce.py" \
        	-file url_pv_sort \
        	-file getURLCoocMap.py \
        	-file getURLCoocReduce.py \
        	-input "${inputdir}" \
        	-output "${outputdir}"
	hadoop fs -text ${outputdir}/part* > url_cooc_${mydate}
} 
getURLCooc;
