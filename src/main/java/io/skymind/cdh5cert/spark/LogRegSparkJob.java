package io.skymind.cdh5cert.spark;

//import RDD;

import org.apache.commons.lang.time.StopWatch;
import org.apache.hadoop.fs.FileSystem;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.mllib.classification.LogisticRegressionModel;
import org.apache.spark.mllib.classification.LogisticRegressionWithLBFGS;
import org.apache.spark.mllib.linalg.DenseVector;
import org.apache.spark.mllib.linalg.Vectors;
import org.apache.spark.mllib.regression.LabeledPoint;
import org.apache.spark.mllib.util.MLUtils;
import org.apache.spark.rdd.RDD;

import scala.reflect.ClassManifestFactory;

public class LogRegSparkJob {
	
	public static String getLabel(String SVMLightRecord) {
		
    	String work = SVMLightRecord.trim();
    	int firstSpaceIndex = work.indexOf(' ');
    	String label = work.substring( 0, firstSpaceIndex );
    	
    	return label.trim();
		
		
	}

	public static String getSVMLightDataColumnsFromRawRecord( String SVMLightRecord ) {
    	
    	String work = SVMLightRecord.trim();
    	int firstSpaceIndex = work.indexOf(' ');
    	String newRecord = work.substring(firstSpaceIndex, work.length());
    	
    	return newRecord.trim();

		
	}
	
	
	
	public static DenseVector convertSVMLightTo_Dense_Vector(String rawLine, int size) {
		
		//Vector sv = Vectors.sparse(3, new int[] {0, 2}, new double[] {1.0, 3.0});
		/*
 SparseVector(int size,
            int[] indices,
            double[] values)
		 */
		
		//System.out.println( "line: " + rawLine );
		
		String[] parts = rawLine.trim().split(" ");
		//System.out.println( "part count: " + parts.length);
		//int[] indicies = new int[ size ]; //[ parts.length - 1 ];
		double[] values = new double[ size ]; //[ parts.length - 1 ];
		
		// skip the label
		//for ( int x = 1; x <  parts.length; x++ ) {
		
		//System.out.println( "Label: " + parts[ 0 ] );
		
		int currentPartsIndex = 1;
		
		for ( int x = 1; x < size + 1; x++ ) {
			
			
			
			String[] indexValueParts = parts[ currentPartsIndex ].split(":");
			int parsedIndex = Integer.parseInt( indexValueParts[ 0 ] );
			if (x == parsedIndex) {

				values[ x - 1 ] = Double.parseDouble(indexValueParts[ 1 ]);

				currentPartsIndex++;
				
			} else {

				values[ x - 1 ] = 0; //Double.parseDouble(indexValueParts[ 1 ]);

			}
			
			//System.out.println( " x = " + x + " -> " + values[ x - 1 ] + ", "+ parts[ x ]);
			
			
		}
		
		// Vectors.dense(1.0, 0.0, 3.0)
		//return new SparseVector(size, indicies, values);
		return (DenseVector) Vectors.dense( values );
		
	}

    public static void main( String[] args) throws Exception {
    	
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
        JavaSparkContext sc = new JavaSparkContext(new SparkConf().setAppName("Skymind_LogisticRegression_Cert_TestJob"));
        
        // training data location
        String hdfsFilesToTrainModel = hdfsPathString; //"hdfs:///user/cloudera/svmlight/*";
        
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
        final int parsedSVMLightFeatureColumns = 4; //Integer.parseInt( svmLightFeatureColumns );
        if (0 == parsedSVMLightFeatureColumns) {
        //	System.err.println("Bad SVMLight feature count: " + svmLightFeatureColumns);
        }
        
        //String hdfsPathToSaveModel = args[ 4 ];
            	
        
        RDD<LabeledPoint> training_svmLight_data_rdd = MLUtils.loadLibSVMFile( sc.sc(), hdfsFilesToTrainModel, true, parsedSVMLightFeatureColumns );
        
        
        /*
        JavaRDD<String> rawAeolipileRecords_trainingRecords = sc.textFile( hdfsFilesToTrainModel );
        
        
        
        JavaRDD< LabeledPoint > trainSVMLightRecords = rawAeolipileRecords_trainingRecords.map(new Function< String, LabeledPoint >() {
            @Override
            public LabeledPoint call(String rawRecordString) throws Exception {
                //Vector features = v1.features();
                //Vector normalized = scalarModel.transform(features);
                //return new LabeledPoint( v1.label(), v1.features() );
            	
            	String[] parts = rawRecordString
            	
            	
            	//String custID = SVMLightUtils.getUniqueIDFromAeolipileRecord( rawRecordString );
            	String svmLight = SVMLightUtils.getSVMLightRecordFromAeolipileRecord( rawRecordString );
            	String label = SVMLightUtils.getLabel( svmLight );
            	double dLabel = Double.parseDouble( label );
            	
            	//Vector svmLightVector = convertSVMLightToVector(svmLight, parsedSVMLightFeatureColumns);
            	Vector svmLightVector = convertSVMLightTo_Dense_Vector( svmLight, parsedSVMLightFeatureColumns );
            	
            	
            
                //return new Tuple2<String, String>( custID, (max + "") );
            	return new LabeledPoint( dLabel, svmLightVector );
            }
        }).cache();    
        
        System.out.println( "\n\n\n" +  trainSVMLightRecords.first() + "\n" );
        */
        JavaRDD<LabeledPoint> training_svmLight_data_JavaRDD = JavaRDD.fromRDD( training_svmLight_data_rdd, ClassManifestFactory.fromClass(LabeledPoint.class) );
        
        System.out.println( "\n\n\n" +  training_svmLight_data_JavaRDD.first() + "\n" );

/*        
        // TODO: change to load text file
        RDD<LabeledPoint> test_svmLight_data_rdd = MLUtils.loadLibSVMFile( sc.sc(), hdfsPathToTestModel, true, parsedSVMLightFeatureColumns );
        
        System.out.println( "\n\n\n" +  test_svmLight_data_rdd.first() + "\n" );
        
        JavaRDD<LabeledPoint> test_svmLight_data_JavaRDD = JavaRDD.fromRDD( test_svmLight_data_rdd, ClassManifestFactory.fromClass(LabeledPoint.class) );
*/
/*
        JavaRDD<String> rawAeolipileRecords_testRecords = sc.textFile( hdfsPathToTestModel );
        
        JavaRDD< LabeledPoint > testSVMLightRecords = rawAeolipileRecords_testRecords.map(new Function< String, LabeledPoint >() {
            @Override
            public LabeledPoint call(String rawRecordString) throws Exception {
                //Vector features = v1.features();
                //Vector normalized = scalarModel.transform(features);
                //return new LabeledPoint( v1.label(), v1.features() );
            	
            //	String custID = SVMLightUtils.getUniqueIDFromAeolipileRecord( rawRecordString );
            	String svmLight = SVMLightUtils.getSVMLightRecordFromAeolipileRecord( rawRecordString );
            	String label = SVMLightUtils.getLabel( svmLight );
            	double dLabel = Double.parseDouble( label );
            	
            	//Vector svmLightVector = convertSVMLightToVector(svmLight, parsedSVMLightFeatureColumns);
            	Vector svmLightVector = SVMLightUtils.convertSVMLightTo_Dense_Vector( svmLight, parsedSVMLightFeatureColumns );
            
                //return new Tuple2<String, String>( custID, (max + "") );
            	return new LabeledPoint( dLabel, svmLightVector );
            }
        }).cache();          
*/        
        
        // Build the model
        
        StopWatch watch = new StopWatch();
        // Run training algorithm to build the model.
        LogisticRegressionWithLBFGS model = new LogisticRegressionWithLBFGS().setNumClasses(3);
        model.optimizer().setMaxNumIterations( numberIterations );
        
        long start = System.currentTimeMillis();
        
        final LogisticRegressionModel model3 = model.run( training_svmLight_data_JavaRDD.rdd() );
        
        long end = System.currentTimeMillis();
        System.out.println("\n\nTime for spark " + Math.abs(end - start) + "\n\n");
        watch.reset();
        
        /*
        // Compute raw scores on the test set.
        JavaRDD<Tuple2<Object, Object>> predictionAndLabels = testSVMLightRecords.map(
                new Function<LabeledPoint, Tuple2<Object, Object>>() {
                    public Tuple2<Object, Object> call(LabeledPoint p) {
                        Double prediction = model3.predict(p.features());
                        return new Tuple2<Object, Object>(prediction, p.label());
                    }
                }
        );
        
        Path outputPath = new Path( hdfsPathToSaveModel );
        
        if ( hdfs.exists( outputPath )) {
        	hdfs.delete( outputPath, true ); 
        } 
        
        model3.save( sc.sc(), hdfsPathToSaveModel );
        
        System.out.println( "Saving Model to: " + hdfsPathToSaveModel );
        
        
        MulticlassMetrics metrics = new MulticlassMetrics(predictionAndLabels.rdd());
        double precision = metrics.fMeasure();
        
        //BinaryClassificationMetrics metrics_binary = new BinaryClassificationMetrics(predictionAndLabels.rdd());
        //metrics_binary.
        
        double[] labels = metrics.labels();
        for ( int x = 0; x < labels.length; x++ ) {
        	
        	System.out.println( "Label: " + labels[x] + ", TP-Rate: " + metrics.truePositiveRate( labels[x]) + ", FP-Rate: " + metrics.falsePositiveRate( labels[x]) );
        	
        }
        */
        
        sc.stop();
        
        //System.out.println("F1 = " + precision);
       // System.out.println("precision = " + metrics.precision() );
       // System.out.println("recall = " + metrics.recall() );
        
        System.out.println( "Skymind Cert Logistic Regression: Job Complete" );
                
       	
    }	
	
}
