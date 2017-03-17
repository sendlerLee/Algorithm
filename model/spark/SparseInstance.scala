package com.ipinyou.test.PLogitBoostAlgo

import org.apache.spark.SparkException
import scala.beans.BeanInfo
import org.apache.spark.mllib.linalg.{Vectors, Vector}
import org.apache.spark.mllib.util.NumericParser
import org.apache.spark.SparkException
import scala.collection.mutable.Set
import scala.collection.mutable.ListBuffer

/**
 * @param label Label for this data point.
 * @param features List of features for this data point.
 */
@BeanInfo
case class SparseInstance(id:String,label:Double, features:List[(String,Double)]) {
  override def toString: String = {
    s"($id,$label,$features)"
  }
}


object SparseInstance {

      def parseWithOutConst(line: String): SparseInstance = {
    	val arr = line.split("\t").toArray
    	if(arr.length > 2){
    	  val id = arr(0).toString()
    	  val label = if(arr(1).toDouble > 0.0 ) 1.0 else 0.0
    	  val features : ListBuffer[(String,Double)] = ListBuffer()
    	  
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
		    	  features.append((featureKey,featureValue))
	    	  	}
    	  }
    	  
    	  SparseInstance(id,label,features.toList)
    	}else{
    	  throw new SparkException(s"Cannot parse string: '$line'.")
    	}
    }
    def parseWithConst(line: String): SparseInstance = {
    	val arr = line.split("\t").toArray
    	if(arr.length > 2){
    	  val id = arr(0).toString()
    	  val label = if(arr(1).toInt > 0 ) 1 else 0
    	  val features : ListBuffer[(String,Double)] = ListBuffer()
    	  
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
		    	  features.append((featureKey,featureValue))
	    	  	}
    	  }
    	  
    	  SparseInstance(id,label,features.toList)
    	}else{
    	  throw new SparkException(s"Cannot parse string: '$line'.")
    	}
    }
      
      def main(args: Array[String]) {
	 	  val s = "Pe5T-_zizy2PxtWFJ-VZI0	0	Adxsddnserror5.wo.com.cn:80802342921611:1	E4EEMq9y0cro:1	362425329:1	Adxsddnserror5.wo.com.cn:80802342921611362425329:1	Adxsddnserror5.wo.com.cn:80802342921611E4EEMq9y0cro:1	E4EEMq9y0cro362425329:1"      
           	val arr = s.split("\t").toArray

    	  val id = arr(0).toString()
    	  val label = if(arr(1).toDouble > 0.0 ) 1.0 else 0.0
    	  val features : ListBuffer[(String,Double)] = ListBuffer()
    	  
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
		    	  features.append((featureKey,featureValue))
	    	  	}
		    	
    	  }
    	  println(features)
    	  SparseInstance(id,label,features.toList)
    	}
}