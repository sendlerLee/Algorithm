#!/bin/bash

if (( $# == 1 ))
then
    end_date=`date "-d +0 day $1" +%Y/%m/%d`
else
    end_date=$(date -d "-1days" +%Y/%m/%d)
fi
#end_date=2014/11/26
yyyy=${end_date:0:4}
mm=${end_date:5:2}
dd=${end_date:8:2}

# Change to base dir
base_dir=$(dirname $0)
cd $base_dir
wd=`pwd`

num=11
train_num=10
total=$num
start_date=$(date -d "${end_date} -${num}days" +%Y/%m/%d)
str_date=$end_date
str_date_list=$(date -d "${end_date} +0days" +%Y%m%d)

function get_time_seq(){
while [ $num -gt 1 ]
do
    num=`expr $num - 1`
    dec_num=`expr $total - $num`
    v_date=`date "-d -$dec_num day $end_date" +%Y/%m/%d`
    str_date=$str_date","$v_date
    str_date_list="$str_date_list $(date -d "${v_date} +0days" +%Y%m%d)"
done
echo "str_date : $str_date"
echo "str_date_list : $str_date_list"

#date1是带斜杠的
python make_date.py "$str_date_list" $train_num > conf/date_win
NBTrain_date1=`cat conf/date_win | awk '$1=="NBTrain1"' | awk '{print $2}'`
NBTrain_date2=`cat conf/date_win | awk '$1=="NBTrain2"' | awk '{print $2}'`
Test_date1=`cat conf/date_win | awk '$1=="Test1"' | awk '{print $2}'`
Test_date2=`cat conf/date_win | awk '$1=="Test2"' | awk '{print $2}'`
}
get_time_seq

pos_date={$str_date}

echo "bash get_user_conv.sh {$str_date}"
bash get_user_conv.sh "{$str_date}"

mkdir -p feature_log
mkdir -p feature_select

function get_feature_company_log(){
    echo $pos_date $1
    hadoop fs -cat /user/liang.ming/online/base_data_othercompany_v6.1/$pos_date/positive/*/$1/p* | awk -F'\t' '{print $2"\t"$5}' \
        > feature_log/feature_${1}_log
    echo "finish generating feature_company_log"
}

trap "exec 6>&-;exec 6<&-;exit 0" 2
tmp_fifofile="/tmp/$.fifo"
echo $tmp_fifofile
mkfifo $tmp_fifofile
exec 6<>$tmp_fifofile
rm $tmp_fifofile
thread=4
for ((i=0;i<$thread;i++));
do
{
    echo >&6
}
done

#手动配置的行业数据，这个行业与每周自动更新的行业id是不同的
taglist=`cat conf/tagdata | grep -v "#" | awk -F'\t' '{print $1}' | sort -u`

function get_feature_tag_log(){
    echo "start generating feature_tag_log"
    echo $pos_date "$1" "$2"
    tag_list="$1"
    tag_company_map_file=$2
    for tag in $tag_list
    do
    {
        rel_company_list=`cat $tag_company_map_file | awk -v tag=$tag -F'\t' '{if($1==tag) print $3}'`
        rm -f feature_log/feature_${tag}_log
        rm -f conv/conv_$tag
        for rel_comp in $rel_company_list
        do
        read -u6
        {
            hadoop fs -cat /user/liang.ming/online/base_data_othercompany_v6.1/$pos_date/positive/*/$rel_comp/p* | awk -F'\t' '{print $2"\t"$5}' \
                >> feature_log/feature_${tag}_log
            cat conv/conv_$rel_comp >> conv/conv_$tag
            echo >&6
        }&
        done
        wait
    }
    done
    echo "finish generating feature_tag_log"
}
get_feature_tag_log "$taglist" "conf/tagdata"

neg_date={$str_date}
function get_feature_0_log(){
    hadoop fs -cat /user/liang.ming/online/base_data_all_v6.1/$neg_date/negative/p* | awk -F'\t' '{print $2"\t"$5}' \
        > feature_log/feature_0_log
    if [ $? -ne 0 ]
    then
        title="/user/liang.ming/online/base_data_all/$neg_date/negative/：dose not exists"
        content="`pwd`/user/liang.ming/online/base_data_all_v6.1/$neg_date/negative/：dose not exists"
        java -cp /data/production/ToolCommon/sendmailudf:/data/production/ToolCommon/sendmailudf/* com.ipinyou.sendmail.Send \
                 "$title" "$content" liang.ming@ipinyou.com
        sh  /data/production/monitor/monitor_information.sh "liang.ming@ipinyou.com" "train_feature_weight_company/run_task.sh" \
            "feature_log/feature_0_log" "/user/liang.ming/online/base_data_all_v6.1/$neg_date/negative/：dose not exists"
        exit 1
    fi
    neg_line_num=`cat feature_log/feature_0_log | wc -l`
    if [ $neg_line_num -lt 1000 ]
    then
        title="line number of feature_log/feature_0_log < 1000"
        content="`pwd`line number of feature_log/feature_0_log < 1000"
        java -cp /data/production/ToolCommon/sendmailudf:/data/production/ToolCommon/sendmailudf/* com.ipinyou.sendmail.Send \
                 "$title" "$content" liang.ming@ipinyou.com
        exit 1
        
    fi
    echo "finish generating feature_0_log"
}
get_feature_0_log

function get_feature_company_select(){
    echo "./urlanalysis_nb feature_log $1 > feature_select/feature_$1_select"
    ./urlanalysis_nb feature_log $1 | awk 'BEGIN{OFS="\t"}$8>0.80{print $0}' | sort -k9gr | head -n 40000 > feature_select/feature_$1_select
}

companyid_tag_list=$taglist
function get_feature_select(){
for companyid in $companyid_tag_list
do
read -u6
{
    get_feature_company_select $companyid
    echo >&6
}&
done
wait
}
get_feature_select

mkdir -p url_model_weight
mkdir -p feature_weight
mkdir -p feature_file
mkdir -p campaign_ext
function generate_url_model_weight(){
    # $1:companyid $2:str_date_list
    echo "./genTrainNBFeature $1 \"$2\""
    ./genTrainNBFeature $1 "$2"
    echo "./genTestNBFeature $1 \"$2\""
    ./genTestNBFeature $1 "$2"

    echo "./logitboost \"campaign_ext/campaign_ext.train.$1\"  \"campaign_ext/campaign_ext.test.$1\" feature_weight/feature_weight_$1"
    ./logitboost "campaign_ext/campaign_ext.train.$1"  "campaign_ext/campaign_ext.test.$1" \
       feature_weight/feature_weight_$1

    echo "python showcatidweight.py $1"
    python showcatidweight.py $1
}

function run_gen_url_model_weight(){
for companyid in $companyid_tag_list
do
read -u6
{
    generate_url_model_weight $companyid "$str_date_list"
    echo >&6
}&
done
wait
}
run_gen_url_model_weight
echo "finish generating feature weight"
exit 1

exec 6>&-
exec 6<&-

