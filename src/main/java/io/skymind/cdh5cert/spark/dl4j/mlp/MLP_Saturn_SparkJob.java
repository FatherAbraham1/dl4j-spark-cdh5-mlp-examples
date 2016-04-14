package io.skymind.cdh5cert.spark.dl4j.mlp;

import org.apache.commons.lang.time.StopWatch;
import org.apache.hadoop.fs.FileSystem;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.mllib.classification.LogisticRegressionModel;
import org.apache.spark.mllib.classification.LogisticRegressionWithLBFGS;
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics;
import org.apache.spark.mllib.evaluation.MulticlassMetrics;
import org.apache.spark.mllib.linalg.DenseVector;
import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.mllib.linalg.Vectors;
import org.apache.spark.mllib.regression.LabeledPoint;
import org.apache.spark.mllib.util.MLUtils;
import org.apache.spark.rdd.RDD;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.Function;
import org.canova.api.records.reader.RecordReader;
import org.canova.api.records.reader.impl.CSVRecordReader;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.api.IterationListener;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.spark.impl.multilayer.SparkDl4jMultiLayer;
import org.deeplearning4j.spark.util.MLLibUtil;
import org.nd4j.linalg.factory.Nd4j;
//import org.nd4j.linalg.io.ClassPathResource;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;

import scala.Tuple2;
import scala.reflect.ClassManifestFactory;

import org.deeplearning4j.datasets.iterator.DataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.api.IterationListener;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction;

import org.apache.log4j.Logger;
import org.apache.log4j.Level;




public class MLP_Saturn_SparkJob {

	

	public static String getLabel(String SVMLightRecord) {
		
    	String work = SVMLightRecord.trim();
    	int firstSpaceIndex = work.indexOf(' ');
    	String label = work.substring( 0, firstSpaceIndex );
    	
    	return label.trim();
		
		
	}
/*
	public static String getSVMLightDataColumnsFromRawRecord( String SVMLightRecord ) {
    	
    	String work = SVMLightRecord.trim();
    	int firstSpaceIndex = work.indexOf(' ');
    	String newRecord = work.substring(firstSpaceIndex, work.length());
    	
    	return newRecord.trim();

		
	}
	
	*/
	
	public static DenseVector convert_CSV_To_Dense_Vector(String rawLine, int size) {
		
		//Vector sv = Vectors.sparse(3, new int[] {0, 2}, new double[] {1.0, 3.0});
		/*
 SparseVector(int size,
            int[] indices,
            double[] values)
		 */
		
		//System.out.println( "line: " + rawLine );
		
		String[] parts = rawLine.trim().split(",");
		//System.out.println( "part count: " + parts.length);
		//int[] indicies = new int[ size ]; //[ parts.length - 1 ];
		double[] values = new double[ size ]; //[ parts.length - 1 ];
		
		// skip the label
		//for ( int x = 1; x <  parts.length; x++ ) {
		
		//System.out.println( "Label: " + parts[ 0 ] );
		
	//	int currentPartsIndex = 1;
		
		for ( int x = 1; x < size + 1; x++ ) {
			
			
			
			//String[] indexValueParts = parts[ currentPartsIndex ].split(":");
			double parsedValue = Double.parseDouble( parts[ x ] );
			values[ x - 1 ] = parsedValue;

			
			//System.out.println( " x = " + x + " -> " + values[ x - 1 ] + ", "+ parts[ x ]);
			
			
		}
		
		// Vectors.dense(1.0, 0.0, 3.0)
		//return new SparseVector(size, indicies, values);
		return (DenseVector) Vectors.dense( values );
		
	}
	
    /**Fit the data, splitting into smaller data subsets if necessary. This allows large {@code JavaRDD<DataSet>}s)
     * to be trained as a set of smaller steps instead of all together.<br>
     * Using this method, training progresses as follows:<br>
     * train on {@code examplesPerFit} examples -> average parameters -> train on {@code examplesPerFit} -> average
     * parameters etc until entire data set has been processed<br>
     * <em>Note</em>: The actual number of splits for the input data is based on rounding up.<br>
     * Suppose {@code examplesPerFit}=1000, with {@code rdd.count()}=1200. Then, we round up to 2000 examples, and the
     * network will then be fit in two steps (as 2000/1000=2), with 1200/2=600 examples at each step. These 600 examples
     * will then be distributed approximately equally (no guarantees) amongst each executor/core for training.
     *
     * @param rdd Data to train on
     * @param examplesPerFit Number of examples to learn on (between averaging) across all executors. For example, if set to
     *                       1000 and rdd.count() == 10k, then we do 10 sets of learning, each on 1000 examples.
     *                       To use all examples, set maxExamplesPerFit to Integer.MAX_VALUE
     * @param totalExamples total number of examples in the data RDD
     * @param numPartitions number of partitions to divide the data in to. For best results, this should be equal to the
     *                      number of executors
     * @return Trained network
     */
    public static MultiLayerNetwork fitDataSet(SparkDl4jMultiLayer sparkNetwork, JavaRDD<DataSet> rdd, int examplesPerFit, int totalExamples, int numPartitions ){
        
    	int nSplits;
    	MultiLayerNetwork outNetwork = null;
        
    	if(examplesPerFit == Integer.MAX_VALUE || examplesPerFit >= totalExamples ) nSplits = 1;
        else {
            if(totalExamples%examplesPerFit==0){
                nSplits = (totalExamples / examplesPerFit);
            } else {
                nSplits = (totalExamples/ examplesPerFit) + 1;
            }
        }

        if(nSplits == 1){
        	outNetwork = sparkNetwork.fitDataSet( rdd );
        } else {
            double[] splitWeights = new double[nSplits];
            
            for( int i=0; i<nSplits; i++ ) {
            	splitWeights[i] = 1.0 / nSplits;
            }
            
            JavaRDD<DataSet>[] subsets = rdd.randomSplit(splitWeights);
            
            for( int i=0; i<subsets.length; i++ ){
                System.out.println("Initiating distributed training of subset " + (i + 1) + " of " + subsets.length);
                JavaRDD<DataSet> next = subsets[i].repartition(numPartitions);
                outNetwork = sparkNetwork.fitDataSet(next);
                //sparkNetwork.fit
            }
        }
        return outNetwork;
    }	

    public static MultiLayerNetwork fitSplits(JavaSparkContext sc, SparkDl4jMultiLayer sparkNetwork, JavaRDD<LabeledPoint> rdd, int examplesPerFit, int totalExamples, int numPartitions ){
        
    	int nSplits;
    	MultiLayerNetwork outNetwork = null;
        
    	if(examplesPerFit == Integer.MAX_VALUE || examplesPerFit >= totalExamples ) {
    		nSplits = 1;
    	} else {
            if(totalExamples%examplesPerFit==0){
                nSplits = (totalExamples / examplesPerFit);
            } else {
                nSplits = (totalExamples/ examplesPerFit) + 1;
            }
        }
    	
    	System.out.println( "Number Splits: " + nSplits );

        if(nSplits == 1){
        	outNetwork = sparkNetwork.fit( rdd, examplesPerFit );
        } else {
            double[] splitWeights = new double[nSplits];
            
            for( int i=0; i<nSplits; i++ ) {
            	splitWeights[i] = 1.0 / nSplits;
            }
            
            JavaRDD<LabeledPoint>[] subsets = rdd.randomSplit(splitWeights);
            
            System.out.println("\n\nexamplesPerFit: " + examplesPerFit );
            System.out.println("totalExamples: " + totalExamples );
            System.out.println("numPartitions: " + numPartitions );
            
            for( int i=0; i<subsets.length; i++ ){
                System.out.println("Initiating distributed training of subset " + (i + 1) + " of " + subsets.length);
                JavaRDD<LabeledPoint> next = subsets[ i ].repartition(numPartitions);
                
                //System.out.println( "Subset: " + subsets[ i ].count() );
                System.out.println( "Repartitioned RDD Subset Size: " + next.count() );
                System.out.println( "RDD Partition Count: " + next.partitions().size() );
                
                // TODO: match up executors / cores / partitions here --- jump count per fit!
                
                // outNetwork = sparkNetwork.fit(next, (int)next.count());
                outNetwork = sparkNetwork.fit( sc, next );
                
                //sparkNetwork.fit
            }
        }
        return outNetwork;
    }    
    
    public static void main( String[] args) throws Exception {
    	
        Logger.getLogger("org").setLevel(Level.ERROR);
        Logger.getLogger("akka").setLevel(Level.WARN);

    	
        //Nd4j.MAX_SLICES_TO_PRINT = -1;
        //Nd4j.MAX_ELEMENTS_PER_SLICE = -1;
        Nd4j.ENFORCE_NUMERICAL_STABILITY = true;    	
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
        JavaSparkContext sc = new JavaSparkContext(new SparkConf().setAppName("Skymind_MLP_Saturn_Cert_TestJob"));
        
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
        
        JavaRDD< LabeledPoint > trainingRecords = rawCSVRecords_trainingRecords.map(new Function< String, LabeledPoint >() {
            @Override
            public LabeledPoint call(String rawRecordString) throws Exception {
                //Vector features = v1.features();
                //Vector normalized = scalarModel.transform(features);
                //return new LabeledPoint( v1.label(), v1.features() );
            	
            	String[] parts = rawRecordString.split(",");
            	
            	
            	//String custID = SVMLightUtils.getUniqueIDFromAeolipileRecord( rawRecordString );
            	//String svmLight = SVMLightUtils.getSVMLightRecordFromAeolipileRecord( rawRecordString );
            	String label = parts[ 0 ]; //SVMLightUtils.getLabel( svmLight );
            	double dLabel = Double.parseDouble( label );
            	
            	//Vector svmLightVector = convertSVMLightToVector(svmLight, parsedSVMLightFeatureColumns);
            	Vector csvVector = convert_CSV_To_Dense_Vector( rawRecordString, parsedSVMLightFeatureColumns );
            	
            	
            
                //return new Tuple2<String, String>( custID, (max + "") );
            	return new LabeledPoint( dLabel, csvVector );
            }
        }).cache();    
        
        System.out.println( "\n\nTraining Record:\n" +  trainingRecords.first() + "\n" );
        
        
        JavaRDD<String> rawCSVRecords_testRecords = sc.textFile( hdfsFilesToTestModel );
        
        JavaRDD< LabeledPoint > testRecords = rawCSVRecords_testRecords.map(new Function< String, LabeledPoint >() {
            @Override
            public LabeledPoint call(String rawRecordString) throws Exception {
                //Vector features = v1.features();
                //Vector normalized = scalarModel.transform(features);
                //return new LabeledPoint( v1.label(), v1.features() );
            	
            	String[] parts = rawRecordString.split(",");
            	
            	
            	//String custID = SVMLightUtils.getUniqueIDFromAeolipileRecord( rawRecordString );
            	//String svmLight = SVMLightUtils.getSVMLightRecordFromAeolipileRecord( rawRecordString );
            	String label = parts[ 0 ]; //SVMLightUtils.getLabel( svmLight );
            	double dLabel = Double.parseDouble( label );
            	
            	//Vector svmLightVector = convertSVMLightToVector(svmLight, parsedSVMLightFeatureColumns);
            	Vector csvVector = convert_CSV_To_Dense_Vector( rawRecordString, parsedSVMLightFeatureColumns );
            	
            	
            
                //return new Tuple2<String, String>( custID, (max + "") );
            	return new LabeledPoint( dLabel, csvVector );
            }
        }).cache();    
        
        System.out.println( "\n\nTest Record: \n" +  testRecords.first() + "\n" );        
        
        

        
        // Build the model
        
        StopWatch watch = new StopWatch();
        long start = System.currentTimeMillis();
        /*
        // Run training algorithm to build the model.
        LogisticRegressionWithLBFGS model = new LogisticRegressionWithLBFGS().setNumClasses(3);
        model.optimizer().setMaxNumIterations( numberIterations );
        
        
        
        final LogisticRegressionModel model3 = model.run( training_svmLight_data_JavaRDD.rdd() );
        */
        
        MultiLayerConfiguration dl4j_network_conf = new NeuralNetConfiguration.Builder()
        .seed(seed)
        .iterations(iterations)
        .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
        .learningRate(learningRate)
        .updater(Updater.NESTEROVS).momentum(0.9)
        .list(2)
        .layer(0, new DenseLayer.Builder().nIn(numInputs).nOut(numHiddenNodes)
                .weightInit(WeightInit.XAVIER)
                .activation("relu")
                .build())
        .layer(1, new OutputLayer.Builder(LossFunction.NEGATIVELOGLIKELIHOOD)
                .weightInit(WeightInit.XAVIER)
                .activation("sigmoid").weightInit(WeightInit.XAVIER)
                .nIn(numHiddenNodes).nOut(numOutputs).build())
        .pretrain(false).backprop(true).build();


        MultiLayerNetwork networkModel = new MultiLayerNetwork( dl4j_network_conf );
        networkModel.init();
        networkModel.setUpdater( null );
        networkModel.setListeners(Collections.singletonList((IterationListener) new ScoreIterationListener(listenerFreq)));
        
        SparkDl4jMultiLayer sparkNetwork = new SparkDl4jMultiLayer( sc, networkModel );
        
        
        //SparkDl4jMultiLayer trainLayer = new SparkDl4jMultiLayer( sc.sc(), dl4j_network_conf );
        
        //model.setListeners(Collections.singletonList((IterationListener) new ScoreIterationListener(listenerFreq)));
        
        //trainingRecords
        
        long totalRecs = trainingRecords.count();
        
        System.out.println( "\n\nNumber records: " + totalRecs );
        System.out.println( "Number partitions: " + trainingRecords.partitions().size() + "\n\n" );
        
        JavaRDD< LabeledPoint > trainingRecordsRePartitioned = trainingRecords.repartition( numberDL4JFitPartitions );
        /*
        JavaRDD<DataSet> trainingRecordsDataset = MLLibUtil.fromLabeledPoint(trainingRecordsRePartitioned, 2, 200);
        
        System.out.println( "Repartitioned: partitions: " + trainingRecordsDataset.partitions().size() + "\n\n" );
        
        System.out.println( "DataSet Records: " + trainingRecordsDataset.count() + "\n\n" );
        
        System.out.println( "First Dataset: " + trainingRecordsDataset.first().get(0) );
        */
        
        // executor count * max recs per executor
        int miniBatchSize = numberDL4JFitPartitions * maxRecordsPerFitPerWorker;
        //trainingRecords.re
        
        MultiLayerNetwork inProcessNetwork = null;
        
        //fit on the training set
        for ( int n = 0; n < nEpochs; n++) {
        	System.out.println( "\n\n> ---------------------- Epoch" + n + " ! ------------------- \n\n" ); 
        	//inProcessNetwork = sparkNetwork.fit( sc, trainingRecordsRePartitioned );
        //	sparkNetwork.fi
        	//inProcessNetwork = fitDataSet( sparkNetwork, trainingRecordsDataset, 200, (int)totalRecs, numberDL4JFitPartitions );
        	
        	
        	
        	inProcessNetwork = fitSplits( sc, sparkNetwork, trainingRecordsRePartitioned, miniBatchSize, (int)totalRecs, numberDL4JFitPartitions );
        	
        }
        
        final SparkDl4jMultiLayer trainedNetworkWrapper = new SparkDl4jMultiLayer( sc.sc(), inProcessNetwork );
        
        System.out.println( "\n\n> ---------------------- Training Complete! ------------------- \n\n" ); 
        
        long end = System.currentTimeMillis();
        System.out.println("\n\nTime for training " + Math.abs(end - start) + "\n\n");
        watch.reset();
        
        
        System.out.println( "\n\n> ---------------------- [ Calculating F1 Score ] ------------------- \n\n\n" ); 
        
     // Compute raw scores on the test set.
        JavaRDD<Tuple2<Object, Object>> predictionAndLabels = testRecords.map(
                new Function<LabeledPoint, Tuple2<Object, Object>>() {
                    public Tuple2<Object, Object> call(LabeledPoint p) {
                        Vector prediction = trainedNetworkWrapper.predict(p.features());
                        double max = 0;
                        double idx = 0;
                        for(int i = 0; i < prediction.size(); i++) {
                            if(prediction.apply(i) > max) {
                                idx = i;
                                max = prediction.apply(i);
                            }
                        }

                        return new Tuple2<Object, Object>(idx, p.label());
                    }
                }
        );        
        
     
        
        



//     RDD<Tuple2<Double, Double>> predictionsRDD = JavaRDD.toRDD( predictionAndLabels );

        // Get evaluation metrics.
        MulticlassMetrics multi_class_metrics = new MulticlassMetrics( JavaRDD.toRDD( predictionAndLabels ));
        
        double precision = multi_class_metrics.fMeasure();
        double recall = multi_class_metrics.recall();
        
        BinaryClassificationMetrics metrics = new BinaryClassificationMetrics(JavaRDD.toRDD( predictionAndLabels ));
        double auROC = metrics.areaUnderROC();  
        	    
        System.out.println( "\n> AUC: " + auROC + "" );
        System.out.println( "\n> Precision: " + precision + "" );
        System.out.println( "\n> Recall: " + recall + "" );
               
        
        
       
        
        
        sc.stop();
        
        //System.out.println("F1 = " + precision);
       // System.out.println("precision = " + metrics.precision() );
       // System.out.println("recall = " + metrics.recall() );
        
        System.out.println( "Skymind Cert > MLP-Saturn: Job Complete" );
          
       	
    }		
	
	
}
