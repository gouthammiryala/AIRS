package com.ndsu.AIRS;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.ml.evaluation.RegressionEvaluator;
import org.apache.spark.ml.recommendation.ALS;
import org.apache.spark.ml.recommendation.ALSModel;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;



public class TrainALS {
	public static void main(String[] args) throws IOException {

		
		if (args.length == 1 && args[0].equalsIgnoreCase("help"))
		{
			System.out.println("Please provide the following arguments: ");
			System.out.println("1. Maximum number of iterations to run the ALS Algorithm");
			System.out.println("2. Regularisation Parameter for ALS Algorithm");
			System.out.println("3. Filepath for testing and training to find best RMSE");
			System.exit(0);
		}
		else if (args.length<3)
		{
			String exception = "Please provide the following arguments: "
					+"/n1. Maximum number of iterations to run the ALS Algorithm"
					+"/n2. Regularisation Parameter for ALS Algorithm"
					+"/n3. Filepath for testing and training to find best RMSE";
			throw new IllegalArgumentException(exception);
		}
		
		System.out.println("Input: Number of iterations: "+args[0]
				+"\nRegularisation parameters:"+args[1]
				+"\nFilePath: "+args[2]);
		List<Integer> maxIterations = new ArrayList<Integer>();
		List<Double> regParams = new ArrayList<Double>();
		
		for (String maxIter : args[0].split(",")){
			maxIterations.add(Integer.parseInt(maxIter));
		}
		for (String regParam : args[1].split(",")){
			regParams.add(Double.parseDouble(regParam));
		}
		
		String filePath = args[2];

		SparkSession spark = SparkSession
				.builder()
				.appName("JavaALSExample").master("local[4]")
				.getOrCreate();

		JavaRDD<Rating> ratingsRDD = spark
				.read().textFile(filePath).javaRDD()
				.map(new Function<String, Rating>() {
					public Rating call(String str) {
						return Rating.parseRating(str);
					}
				});

		Dataset<Row> ratings = spark.createDataFrame(ratingsRDD, Rating.class);
		Dataset<Row>[] splits = ratings.randomSplit(new double[]{0.6, 0.4});
		Dataset<Row> training = splits[0];
		Dataset<Row> test = splits[1];

		String s = "";

		ALSModel bestModel = null;
		double rmse = Double.MAX_VALUE;

		
		

		for (int maxIteration : maxIterations)
		{
			for (double regParam : regParams){
				ALS als = new ALS()
						.setMaxIter(maxIteration)
						.setRegParam(regParam)
						.setUserCol("userId")
						.setItemCol("movieId")
						.setRatingCol("rating");


				ALSModel model = als.fit(training);

//				model.setColdStartStrategy("drop");    		

				// Evaluate the model by computing the RMSE on the test data
				Dataset<Row> predictions = model.transform(test);   		

				//	    				System.out.println("******************************************");

				RegressionEvaluator evaluator = new RegressionEvaluator()
						.setMetricName("rmse")
						.setLabelCol("rating")
						.setPredictionCol("prediction");

				double localrmse = evaluator.evaluate(predictions);
				s += "maxIteration: "+ maxIteration + ", regularisation parameter: "+ regParam + " RMSE: " + localrmse+"\n";
				model.save("model/ALSModel"+"_"+maxIteration+"_"+regParam);
				if (rmse > localrmse)
				{
					rmse = localrmse;
					//bestModel = model;
				}
			}	    			
		}

		

		FileWriter f = new FileWriter("RMSEStats.txt");
		f.write(s);
		f.flush();
		f.close();


		System.out.println("Best RMSE is  = " + rmse);
		spark.stop();
	}
}
