package com.ipinyou.test.PLogitBoostAlgo
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.SparkContext._
import org.apache.spark.mllib.util.MLUtils
import scala.collection.mutable.Set
import scala.collection.mutable.Map
import scala.collection.mutable.ListBuffer
import java.util.Random
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import breeze.linalg.{Vector, DenseVector, squaredDistance}
import scala.collection.mutable.ArrayBuffer

object DataProcess {

	  def main(args: Array[String]) {
	    makeTrainSet(args)
	  } 
      
     def makeTrainSet(args: Array[String]) ={
       
       	if (args.length < 3) {
	      System.err.println("Usage: Parallel Logit Boost Algorithm DataProcessing: <origin_trainSet_hdfsPath_in> <tmpFeatureSet_hdfsPath_out> <tmpTrainInstances_hdfsPath_out> ")
	      System.exit(1)
	    }
       	println("-----------------DATA PROCESS-----------------")
       	println("Usage: Parallel Logit Boost Algorithm DataProcessing: <origin_trainSet_hdfsPath_in> <tmpFeatureSet_hdfsPath_out> <tmpTrainInstances_hdfsPath_out> ")

       	val trainSetHDFSPath = args(0)    
       	val featureSetPath = args(1) //"/user/xijun.piao/test/parallelBoostingAlgo/tmp_feature_out/1"
       	val tmpInstances_hdfsPath = args(2)//"/user/xijun.piao/test/parallelBoostingAlgo/tmp_train_plus/1"
       		    
       	val sparkConf = new SparkConf().setAppName("Parallel Logit Boost Algorithm DataProcessing")

	    val sc = new SparkContext(sparkConf)
	    //1. load row training data
	    val lines = sc.textFile(trainSetHDFSPath)
	    val data = lines.map(SparseInstance.parseWithOutConst _)
	    val originInstaceSize = data.count
	    
	    
	    //2. prepare
	    //find all useful features & make stringId to idx Map
	    val stringId2Idx_Map_Rdd = data.flatMap(instance => {
	         instance.features.map(f => f._1).toList
	     }).distinct.zipWithIndex

	    stringId2Idx_Map_Rdd.map(x=>  x._1+" "+x._2 ).saveAsTextFile(featureSetPath)
    
	    val weightVectorRdd = sc.textFile(featureSetPath).map(
       	  line => {
       	    def arr = line.split(" ")
       	    (arr(0),(arr(1).toInt))
       	  }
       	)
       	weightVectorRdd.cache
	    val weightVectorTotalSize = weightVectorRdd.count.toInt //withOutConst
	    val constBiasIndex = weightVectorTotalSize
	    
		// train data	  

       	val rowTrainData = data.zipWithIndex.flatMap(instanceWithIndex => {
		  //feature_name (instance_index, label, feature_x_value instance_id)
		  var res = ListBuffer[(String,(Long,Double,Double,String))]()
		  for( s <- instanceWithIndex._1.features){
		    res.append((s._1, (instanceWithIndex._2, instanceWithIndex._1.label, s._2, instanceWithIndex._1.id) ))
		  }
		  res.toList
		}).join(weightVectorRdd).map{
		  case (key, ((instanceIndex, label, featureValue, instanceId), (featureIdxId))) => {
		        val value = (featureIdxId.toLong, featureValue)
			    ( (instanceIndex,label,instanceId), ArrayBuffer(value) )
		  }
		}
       			
       	
	    val tmpTrainData = rowTrainData.reduceByKey{
	      case (featureIdxWithValue1, featureIdxWithValue2) =>{
	        for ( idx <- featureIdxWithValue2){
	          //featureIdxId1 += idx
	          featureIdxWithValue1.append(idx)
	        }
	        (featureIdxWithValue1)
	      }
	    }
	    //filter invalid data
	    val train_data = tmpTrainData.filter{case (instanceID_label,features) => if(features.size > 0) true else false}.map{
	      case ((instanceIndexId,label,instanceId),features) => {
	        features.append((constBiasIndex,1.0))
	        val sorted_features = features.sortWith((x,y)=> (x._1 <= y._1))
	        (instanceId,label,sorted_features)
	      }
	    }
	    val filteredInstanceSize = train_data.count
	    train_data.map{
	      case (instanceId,label,features) => 
	        var result = "" //instanceId + " "
	        if(features.size > 0){
	          result += label.toString
	          if(features.size > 1){
		        for(i <- 0 until features.size ){
		        	result += " " + features(i)._1.toString + ":" + features(i)._2.toString 
		        }
	          }

	        }
	        result
	     }.saveAsTextFile(tmpInstances_hdfsPath)
	    
	    println("--------*-----------")
	    println("originInstaceSize:" + originInstaceSize)
	    println("filteredInstanceSize:" + filteredInstanceSize)
	    println("featureVectorSize:" + weightVectorTotalSize)
     }
}