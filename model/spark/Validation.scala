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
import scala.util.control.Breaks._
import org.apache.spark.SparkException

object Validation {
  
	def main(args: Array[String]) {

	    validating(args)
	}
	
	def validating(args:Array[String]) = {
	  
	    if (args.length < 2) {
	      System.err.println("Usage: Parallel Logit Boost Algorithm Validating: <featureWeight_hdfsPath> <validationSet_hdfsPath> <output_hdfs_path>")
	      System.exit(1)
	    }
        println("-----------------VALIDATION-----------------")
        println("Usage: Parallel Logit Boost Algorithm Validating: <featureWeight_hdfsPath> <validationSet_hdfsPath> <output_hdfs_path>")
	    val featureWeightHdfsPath = args(0)
	    val validationInstanceHdfsPath = args(1)
	    val w = 0.0
	    
       	val sparkConf = new SparkConf().setAppName("Parallel Logit Boost Algorithm Validating")
       	sparkConf.set("spark.driver.maxResultSize", "3g")
	    val sc = new SparkContext(sparkConf)

       	val weightDic = sc.textFile(featureWeightHdfsPath).map(
       	  line => {
       	     def arr = line.split(" ")
       	     (arr(0),arr(2).toDouble)
       	  }
       	)
       	
       	weightDic.cache
       	
       	val const_arr = weightDic.filter(f => (if("const".equalsIgnoreCase(f._1)) true else false)).collect
       	val const = if(const_arr.length > 0) const_arr(0)._2 else 0.0

	    
		//  data
       	val validate_data = sc.textFile(validationInstanceHdfsPath).zipWithIndex.flatMap{
	      case (line,idx) => 
       		  val arr = line.split("\t").toArray
       		  val id_features : ListBuffer[(String,(Long,Double,Double,String))] = ListBuffer()
	    	  val id = arr(0).toString()
   			  val label = if(arr(1).toDouble > 0.0 ) 1.0 else 0.0
	    	  for (i <- 2 until (arr.length) ) {
			    	val kvPair = arr(i).toString().split(":")
			    	if(kvPair.length > 1 && (!"const".equalsIgnoreCase(kvPair(0)))){
			    	  val featureValue = kvPair(kvPair.length-1).toDouble
			    	  var featureKey = kvPair(0).toString()
			    	  if(kvPair.length > 2){
			    	    for( j <- 1 until (kvPair.length-1)){
			    	      featureKey+=":"+kvPair(j)
			    	    }
			    	  }
			    	  id_features.append((featureKey,(idx,label,featureValue,id)))
		    	  	}
	    	  }
       		  id_features.toList
       	}
  
		val tmpPredAndLabelTest = validate_data.join(weightDic).map{
		  case (keyFeatureName, ((instanceIdx,label,featureValue,id),  featureWeight)) => {
		        val value = featureValue * featureWeight
			    ( (instanceIdx,label,id), value )
		  }
		}.reduceByKey(_+_).map(instance => (instance._1._3, ((1/(1+ math.exp(-1.0 * (instance._2+const)))).toDouble, instance._1._2) ))
		
		if (args.length >2) {
		  tmpPredAndLabelTest.saveAsTextFile(args(2))
		}
		
		    
		val predAndLabelTest =  tmpPredAndLabelTest.map( id_score_label => ( id_score_label._2._1,id_score_label._2._2 ))
		predAndLabelTest.cache
		
	    val instanceNum = predAndLabelTest.count.toDouble;
 	

       	
	    val squaredError = predAndLabelTest.map(score_label =>{
	        (score_label._2 - score_label._1) * (score_label._2 - score_label._1)
	    }).aggregate(0.0)(_+_, _+_)
	    
	    val tmpllh = predAndLabelTest.map(x =>{
		  val ctr = x._1
		  val label = x._2
		  	if (x._1 != 1.0 && x._1 != 0.0){
		  	  label * math.log(ctr) + (1 - label) * math.log(1 - ctr);
		  	}else{
		  	  0.0
		  	} 
		}).aggregate(0.0)(_+_, _+_)
	    
	    val mse = squaredError/instanceNum;

	    val llh = tmpllh/instanceNum
       	val metrics = new BinaryClassificationMetrics(predAndLabelTest)
		val ROC = metrics.areaUnderROC()
		
	
		
		println("--------Validation RESULT-----------")
		println("ValidateSet Size:" + instanceNum)
	  	println("ROC:" + ROC)
	  	println("MSE:" + mse)
	  	println("LLH:" + llh)
	}



}