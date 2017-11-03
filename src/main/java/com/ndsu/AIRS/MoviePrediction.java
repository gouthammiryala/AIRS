package com.ndsu.AIRS;

import java.util.Arrays;
import java.util.Iterator;
import java.util.regex.Pattern;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.function.FlatMapFunction;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.api.java.function.VoidFunction2;
import org.apache.spark.ml.recommendation.ALSModel;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.api.java.StorageLevels;
import org.apache.spark.streaming.Durations;
import org.apache.spark.streaming.Time;
import org.apache.spark.streaming.api.java.JavaDStream;
import org.apache.spark.streaming.api.java.JavaReceiverInputDStream;
import org.apache.spark.streaming.api.java.JavaStreamingContext;

public final class MoviePrediction {
  private static final Pattern SPACE = Pattern.compile(" ");
  public static void main(String[] args) throws Exception {

    SparkConf sparkConf = new SparkConf().setAppName("MoviePrediction").setMaster("local[4]");
    JavaStreamingContext ssc = new JavaStreamingContext(sparkConf, Durations.seconds(1));

    JavaReceiverInputDStream<String> lines = ssc.socketTextStream(
        "localhost", 9999, StorageLevels.MEMORY_AND_DISK_SER);
    JavaDStream<String> readStream = lines.flatMap(new FlatMapFunction<String, String>() {
      @Override
      public Iterator<String> call(String x) {
        return Arrays.asList(SPACE.split(x)).iterator();
      }
    });
    
	ALSModel model = ALSModel.load("models/ALSModel_20_0.01");

    // Convert RDDs of the words DStream to DataFrame and run SQL query
    readStream.foreachRDD(new VoidFunction2<JavaRDD<String>, Time>() {
      @Override
      public void call(JavaRDD<String> rdd, Time time) {
        SparkSession spark = JavaSparkSessionSingleton.getInstance(rdd.context().getConf());
        // Convert JavaRDD[String] to JavaRDD[bean class] to DataFrame
        JavaRDD<Rating> rowRDD = rdd.map(new Function<String, Rating>() {
          @Override
          public Rating call(String word) {
        	  Rating record = Rating.parseRating(word);
           // record.setWord(word);
            return record;
          }
        });
        Dataset<Row> wordsDataFrame = spark.createDataFrame(rowRDD, Rating.class);

        System.out.println("========= " + time + "=========");
        Dataset<Row> predictions =  model.transform(wordsDataFrame);
      //  predictions.write().csv("output.csv");;
        predictions.show();
      }
    });
    ssc.start();
    ssc.awaitTermination();
  }
}
/** Lazily instantiated singleton instance of SparkSession */
class JavaSparkSessionSingleton {
  private static transient SparkSession instance = null;
  public static SparkSession getInstance(SparkConf sparkConf) {
    if (instance == null) {
      instance = SparkSession
        .builder()
        .config(sparkConf)
        .getOrCreate();
    }
    return instance;
  }
}
