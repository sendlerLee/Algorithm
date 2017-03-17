#!/bin/bash
if (( $# == 1 ))
then
    all_date_list=$1
else
    echo "Usage Error"
    echo "Usage: $0 all_date_list({yyyy/mm/dd,yyyy/mm/dd})"
    exit 1
fi

# Change to base dir
base_dir=$(dirname $0)
cd $base_dir

mkdir -p conv

function get_user_conv(){
    echo $all_date_list
    hadoop fs -cat /user/liang.ming/tmp/lookalike/analysis_day/conv_v6/$all_date_list/cvt/p* | \
      awk 'BEGIN{FS=OFS="\t"}{print $1 > "conv/conv_"$2""}'
}
get_user_conv

