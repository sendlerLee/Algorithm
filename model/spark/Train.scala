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
object Train {

      def main(args: Array[String]) {
	    train(args)
	  }

      def train(args: Array[String]) ={
    
        if (args.length < 3) {
	      System.err.println("Usage: Parallel Logit Boost Algorithm Training: <tmpFeatureSet_hdfsPath_in> <tmpInstances_hdfsPath_in> <featureMap&weight_hdfsPath_out> <round_int> <stepSizePara_double> <regulationPara_double>  ")
	      System.exit(1)
	    }
        println("-----------------TRAIN-----------------")
	    println("Usage: Parallel Logit Boost Algorithm Training: <tmpFeatureSet_hdfsPath_in> <tmpInstances_hdfsPath_in>  <featureMap&weight_hdfsPath_out> <round_int> <stepSizePara_double> <regulationPara_double> ")

	    //params
	    val tmpFeatureSet_hdfsPath = args(0)
	    val tmpInstances_hdfsPath = args(1) 
	    val featureMapWeight_hdfsPath_out = args(2) 
	    
	    println("feature in path:"+tmpFeatureSet_hdfsPath)
	    println("training instance path:"+tmpFeatureSet_hdfsPath)
	    println("feature weight out path:"+featureMapWeight_hdfsPath_out)
	    
	    //T:  number of Rounds
	    println("-----------------params:---------------------------")
	    val T = if (args.length > 3) args(3).toInt else 10
	    println("training will loops T:"+T)
	    //stepSize param
	    val stepSize = if (args.length >4) args(4).toDouble else 0.5  
	    println("stepSize:"+stepSize)
	    
	    val regulationPara = if (args.length > 5) args(5).toDouble else 0.0  
	    println("lamda(regularization):"+regulationPara)
	    
       
       	val sparkConf = new SparkConf().setAppName("Parallel Logit Boost Algorithm Training")
       	sparkConf.set("spark.driver.maxResultSize", "4g")
	    val sc = new SparkContext(sparkConf)
	    //1. load data
	    //vector mapping
       	val weightVectorRdd = sc.textFile(tmpFeatureSet_hdfsPath).map(
       	  line => {
       	    def arr = line.split(" ")
       	    (arr(0),(arr(1).toInt))
       	  }
       	)
       	val weightVectorTotalSize = weightVectorRdd.count.toInt + 1 //  +const
       	println("weightVectorTotalSize:" + weightVectorTotalSize)
       	       	//weight vector
	    val weightVector  = new Array[Double](weightVectorTotalSize.toInt)
	    for(i <- 0 until (weightVector.length) ){
	      weightVector(i) = 0.0
	    }
     	
	    //training data (indexId,label,featrue&values)
       	val train_data = sc.textFile(tmpInstances_hdfsPath).map{
       	  case line => 
	       	  val arr = line.split(" ").toArray
	          val label = arr(0).toDouble
	          val featrue_value_pair_list = new Array[(Int,Double)](arr.length-1)
	          for(i <- 1 until arr.length){
	            val tmp_feature_value_pair = arr(i).toString.split(":")
	            
	            
	            if(tmp_feature_value_pair.length==2){
	              featrue_value_pair_list(i-1)=(tmp_feature_value_pair(0).toInt,tmp_feature_value_pair(1).toDouble)
	            }
	          }
	       	  (label,featrue_value_pair_list)
       	}
       	train_data.cache
       	
       	
       	val instanceNum = train_data.count.toDouble;
       	println("training instance Num:"+instanceNum)
       	val PosCnt =train_data.aggregate(0.0)((x,y)=>(x + y._1), _+_)
        val NegCnt = instanceNum - PosCnt;
		//F≥ı÷µ
		val F0 =  math.log(PosCnt * 1.0 / NegCnt);
		weightVector(weightVector.length-1) = F0
//	    println("F0:" + F0)
	    
	    
	    //3. TRAIN°°LOOP
	    for (x <- 1 to T){
//	    		val predAndLabelTest = train_data.map{
//			       case (features) =>
//			        	 (predictScore(weightVector,features._2),features._1.toDouble)
//			    }
//			    val squaredError = predAndLabelTest.map(score_label =>{
//			        (score_label._2 - score_label._1) * (score_label._2 - score_label._1)
//			    }).aggregate(0.0)(_+_, _+_)
	    		
      	    	//F Q Z ∏¸–¬
			    val F_Q_Z_train = train_data.map(instance=>{
			      var Fi = 0.0
			      for( features <-instance._2){
			        Fi += weightVector(features._1) * features._2
			      }
			      val Pi= 1.0 / (1 + math.exp(-Fi))
			      val Qi = Pi * (1 - Pi)
			      val Zi = instance._1 - Pi
			      (Qi,Zi,Pi,instance)
			    })
			    F_Q_Z_train.cache
			    
			    val numeratorvec_denominatorvec = F_Q_Z_train.flatMap(instance =>{
			      val Qi = instance._1
			      val Zi = instance._2
			      val instace = instance._4._2
			      instace.map( feature_value => (feature_value._1, (feature_value._2 * Zi,feature_value._2 * feature_value._2 * Qi)) ).toList
			    }).reduceByKey((x,y)=>(x._1+y._1,x._2+y._2)).collect().sortWith((x,y) => (x._1 <= y._1)) 
			    
		    
				var tmpNumeratorvecIdx = 0
				for (i <- 0 until weightVector.length){
				   if(numeratorvec_denominatorvec(tmpNumeratorvecIdx)._1==i){
				     weightVector(i) += stepSize * numeratorvec_denominatorvec(tmpNumeratorvecIdx)._2._1 / (numeratorvec_denominatorvec(tmpNumeratorvecIdx)._2._2 + regulationPara);
				     tmpNumeratorvecIdx +=1;
				   }
				}
		    
			    val trainSquaredError = F_Q_Z_train.aggregate(0.0)((x,y)=>(x + (y._4._1 - y._3) * (y._4._1 - y._3)), _+_)
			    val tmpTrainLLH = F_Q_Z_train.aggregate(0.0)((x,y)=>{
			    		if(y._3 != 1 && y._3 != 0){
			    			x + y._4._1 * math.log(y._3) + (1 - y._4._1) * math.log(1 - y._3)
			    		}else{
			    			0.0
			    		}
			    	}, _+_)
			    
			 
			   //info
			    val trainMSE = trainSquaredError/instanceNum;
			    val trainLLH = tmpTrainLLH/instanceNum

			    //llh caculate
			    println("------------------------ROUND "+ x +" ------------------------")
			    //printWeightVector(weightVector:Array[Double])
			  	println("MSE:" + trainMSE)
			  	println("LLH:" + trainLLH)
			 
			 println("--------------------------------------")
	  	}
		val predAndLabelTest = train_data.map{
	       case (features) =>
	        	 (predictScore(weightVector,features._2),features._1.toDouble)
	    }
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
		val trainROC = metrics.areaUnderROC()
		println("--------------FINAL TRAINING RESULT------------------------")
		println("Final AUC:" + trainROC)
		println("Final MSE:" + mse)
		println("Final llh:" + llh)
	    //save feature weight to HDFS
       	val weightVectorRddFinal =  weightVectorRdd.map(x => {
       			x._1+ " " + x._2+" " + weightVector(x._2)
       		}
       	)
       	val constValue = weightVector(weightVector.length-1) //+ F0
       	val constValueRDD = sc.parallelize(List("const "+ (weightVector.length-1).toString + " " + constValue.toString()) )
       	(weightVectorRddFinal union constValueRDD).saveAsTextFile(featureMapWeight_hdfsPath_out)

     }
  	  

	  
	  def predictScore(weightVector:Array[Double],pointFeature:Array[(Int,Double)]):Double = {
	    var tmp_score = 0.0
	    for (feature <-  pointFeature){
	       tmp_score += weightVector(feature._1) * feature._2
	    }
	    
	    1/(1+ math.exp(-1.0 * tmp_score))
	  }
	  
	  def printWeightVector(weightVector:Array[Double]) ={
    		println("--------weightVector*-----------")
		    println(0+":"+ weightVector(0))
		    println(1+":"+ weightVector(1))
		    println(2+":"+ weightVector(2))
		    println(3+":"+ weightVector(3))
		    println(4+":"+ weightVector(4))
		    println(5+":"+ weightVector(5))
		    println(6+":"+ weightVector(6))
		    println(7+":"+ weightVector(7))
		    println(8+":"+ weightVector(8))
		    println(9+":"+ weightVector(9))
		    println(100+":"+ weightVector(100))
		    println(101+":"+ weightVector(101))
		    println(102+":"+ weightVector(102))
		    println(103+":"+ weightVector(103))
		    println(104+":"+ weightVector(104))
		    println(105+":"+ weightVector(105))
		    println(106+":"+ weightVector(106))
		    println(107+":"+ weightVector(107))
		    println(108+":"+ weightVector(108))
		    println(109+":"+ weightVector(109))
		  	println("--------*-----------")

	  }
}