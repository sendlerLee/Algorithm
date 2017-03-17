#!/bin/sh

hdstreaming="/usr/lib/hadoop-mapreduce/hadoop-streaming.jar"

mydate=20150324
trainsection=7
midoutputdir=/user/naiqiang.tan/minhash/megeruserurl/${mydate}
resoutputdir=/user/naiqiang.tan/minhash/urlcooc/${mydate}

function getADVDomain()
{
	year=${mydate:0:4}
	month=${mydate:4:2}
	day=${mydate:6:2}

	match=0
        advdatelog=/user/root/flume/express/${year}/${month}/${day}/*/adv*
	for i in `seq 100` 
	do
		curday=`date -d "$i days ago" '+%Y%m%d'`
		if [ $curday -lt $mydate ];then
			curyear=${mydate:0:4}
			curmonth=${curday:4:2}
			curday=${curday:6:2}

                        newdatelog=/user/root/flume/express/${curyear}/${curmonth}/${curday}/*/adv*
                        advdatelog=$advdatelog,$newdatelog

			match=$((match+1))
		fi

		if [ $match = $trainsection ];then
			break
		fi
	done

        #advdatelog=/user/root/flume/express/${year}/${month}/${day}/*/adv*

	echo $bidunbidlog

	# 要在Map中指定文件名
	outputdir=/user/naiqiang.tan/minhash/advurlcnt/${mydate}
	hadoop fs -rmr $outputdir
	hadoop jar ${hdstreaming} \
		-D mapreduce.job.queuename=mapreduce.important \
        	-D mapred.job.name='UserFeature::MergeUserURL' \
        	-D stream.num.map.output.key.fields=1 \
        	-D mapred.reduce.tasks=50 \
        	-mapper "getADVURLDomainMap.py" \
        	-reducer "getADVURLDomainReduce.py" \
        	-file getADVURLDomainMap.py \
        	-file getADVURLDomainReduce.py \
        	-input "${advdatelog}" \
        	-output "${outputdir}"
} 
#getADVDomain;

function mergeUserURL()
{
	year=${mydate:0:4}
	month=${mydate:4:2}
	day=${mydate:6:2}

	match=0
        advdatelog=/user/root/flume/express/${year}/${month}/${day}/*/adv*
	bidunbidlog=/data/production/url_set_pyid/${year}/${month}/${day}/*/part*
	for i in `seq 100` 
	do
		curday=`date -d "$i days ago" '+%Y%m%d'`
		if [ $curday -lt $mydate ];then
			curyear=${mydate:0:4}
			curmonth=${curday:4:2}
			curday=${curday:6:2}

                        newdatelog=/user/root/flume/express/${curyear}/${curmonth}/${curday}/*/adv*
                        advdatelog=$advdatelog,$newdatelog

			newdatelog=/data/production/url_set_pyid/${curyear}/${curmonth}/${curday}/*/part*
                	bidunbidlog=$bidunbidlog,$newdatelog

			match=$((match+1))
		fi

		if [ $match = $trainsection ];then
			break
		fi
	done

	#bidunbidlog=/data/production/url_set_pyid/${year}/${month}/${day}/*/part*
        #advdatelog=/user/root/flume/express/${year}/${month}/${day}/*/adv*

	echo $bidunbidlog

	# 要在Map中指定文件名
	outputdir=$midoutputdir
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
        	-input "${bidunbidlog},${advdatelog}" \
        	-output "${outputdir}"
} 
mergeUserURL;

function getURLCooc()
{
	inputdir=$midoutputdir
	echo $inputdir

	# 要在Map中指定文件名
	outputdir=$resoutputdir
	hadoop fs -rmr $outputdir

	hadoop jar ${hdstreaming} \
		-D mapreduce.job.queuename=mapreduce.important \
        	-D mapred.job.name='UserFeature::URLCooc' \
        	-D stream.num.map.output.key.fields=1 \
        	-D mapred.reduce.tasks=50 \
        	-mapper "getURLCoocMap.py" \
        	-reducer "getURLCoocReduce.py" \
        	-file url_pv_sort \
        	-file adv_netloc_cnt \
        	-file getURLCoocMap.py \
        	-file getURLCoocReduce.py \
        	-input "${inputdir}" \
        	-output "${outputdir}"
	hadoop fs -text ${outputdir}/part* > url_cooc_${mydate}
} 
getURLCooc;
