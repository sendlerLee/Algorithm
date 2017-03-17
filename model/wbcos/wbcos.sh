# 需要配置参数
# INPUT 输入路径, 数据格式为 pyid \t itemid
# OUTPUT 输出路径, 数据格式为 item1 \t item2 \t 相似度
# LOCAL_SIM 本地相似度结果文件

# min_item_user 商品最小UV 
# min_couser 商品间最少共同用户数
# sim_max 商品最多取多少个相关商品
# NAME JOB名称

INPUT=/user/naiqiang.tan/url_similarity/20150817
OUTPUT=/user/naiqiang.tan/url_similarity_result_temp/20150817
LOCAL_SIM=local_sim
NAME="wcos_spark"

min_item_user=2
min_couser=2
sim_max=30

hadoop fs -rm -r $OUTPUT
/data/spark/bin/spark-submit \
  --driver-library-path /usr/lib/hadoop/lib/native/Linux-amd64-64:/usr/lib/hadoop/lib/native/ \
  --name $NAME \
  --master yarn-client \
  --queue spark \
  --executor-memory 2024m \
  wbcos.py \
  $INPUT \
  $OUTPUT \
  $min_item_user \
  $min_couser \
  $sim_max
 
  #--executor-memory 2024m \
  #--executor-memory 2024m \

#rm $LOCAL_SIM
#hadoop fs -getmerge $OUTPUT $LOCAL_SIM
