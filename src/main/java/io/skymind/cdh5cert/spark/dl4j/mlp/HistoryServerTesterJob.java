package io.skymind.cdh5cert.spark.dl4j.mlp;

import org.apache.hadoop.fs.FileSystem;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;

public class HistoryServerTesterJob {

    public static void main( String[] args) throws Exception {
    	
        //Logger.getLogger("org").setLevel(Level.ERROR);
        //Logger.getLogger("akka").setLevel(Level.WARN);

    	
        //Nd4j.MAX_SLICES_TO_PRINT = -1;
        //Nd4j.MAX_ELEMENTS_PER_SLICE = -1;
        //Nd4j.ENFORCE_NUMERICAL_STABILITY = true;    	
        int iterations = 10;
        int seed = 123;
        int listenerFreq = iterations/5;
        double learningRate = 0.005;
        //Number of epochs (full passes of the data)
        int nEpochs = 10;
        
        int numInputs = 2;
        int numOutputs = 2;
        int numHiddenNodes = 20;
    	
        int numberDL4JFitPartitions = 3;
    	
        org.apache.hadoop.conf.Configuration hadoopConfig = new org.apache.hadoop.conf.Configuration();
        FileSystem hdfs = FileSystem.get(hadoopConfig);
/*
        if (5 != args.length) {
        	System.out.println( "Invalid parameters!" );
        	return;
        }
  */      
        String hdfsPathString = args[0];

        // .setMaster("local[*]")
        JavaSparkContext sc = new JavaSparkContext(new SparkConf().setAppName("HistoryServer_Test_Job"));
        
        //System.out.println( "spark.history.fs.logDirectory: " + sc.getConf().get("spark.history.fs.logDirectory") );
        
        // training data location
        String hdfsFilesToTrainModel = hdfsPathString; //"hdfs:///user/cloudera/svmlight/*";
        
        String hdfsFilesToTestModel = args[ 1 ];
        
        String cliLearningRate = args[ 2 ];
        
        String cliEpochs = args[ 3 ];
        
        learningRate = Double.parseDouble(cliLearningRate);
        
        nEpochs = Integer.parseInt( cliEpochs );
        
        numberDL4JFitPartitions = Integer.parseInt( args[ 4 ].toString() );
        
        int maxRecordsPerFitPerWorker = Integer.parseInt( args[ 5 ]  );
        
        // how we define network
        //String hdfsPathToTestModel = args[1];
        
        // where we want to save the model parameters
        //String numberIterationsCmdLine = args[2];
        int numberIterations = 50; //Integer.parseInt( numberIterationsCmdLine );
        //if (0 == numberIterations) {
        //	numberIterations = 50;
        //}
        
        
        // how many features are in input data
        //String svmLightFeatureColumns = args[3];
        final int parsedSVMLightFeatureColumns = 2; //Integer.parseInt( svmLightFeatureColumns );
        if (0 == parsedSVMLightFeatureColumns) {
        //	System.err.println("Bad SVMLight feature count: " + svmLightFeatureColumns);
        }
        
        //String hdfsPathToSaveModel = args[ 4 ];
            	
        
        //RDD<LabeledPoint> training_svmLight_data_rdd = MLUtils.loadLibSVMFile( sc.sc(), hdfsFilesToTrainModel, true, parsedSVMLightFeatureColumns );
        
        
        
        JavaRDD<String> rawCSVRecords_trainingRecords = sc.textFile( hdfsFilesToTrainModel );
        
        System.out.println( "recs: " + rawCSVRecords_trainingRecords.count() );
       
        
        
        sc.stop();
        
        //System.out.println("F1 = " + precision);
       // System.out.println("precision = " + metrics.precision() );
       // System.out.println("recall = " + metrics.recall() );
        
        System.out.println( "History Server : Job Complete" );
          
       	
    }	
	
}
