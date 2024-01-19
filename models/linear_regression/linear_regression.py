from pyspark.sql import SparkSession
from pyspark.ml.regression import LinearRegression
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.mllib.evaluation import MulticlassMetrics

PREDICTION_DIR = "predictions"
MODEL_DIR = "model"
DATA_DIR = "../../data/preprocessed_data.csv"

def main():
    spark: SparkSession = (SparkSession.builder.appName("Linear Regression Training").getOrCreate())
    logger = spark._jvm.org.apache.log4j.LogManager.getLogger("Linear Regression Training")
    
    FEATURES = ['lag_like_count', 'lag_skip_count', 'lag_playtime_ratio', 
               'lag_1', 'lag_2', 'lag_3', 'lag_4', 'lag_5', 'popularity', 'release_date',
               'duration_ms', 'danceability', 'acousticness', 'instrumentalness']
    TARGET = ['play_count']

    data = spark.read.csv(DATA_DIR, header=True, inferSchema=True)
    
    train_data = data.filter(data['week'] <= 100)
    test_data = data.filter(data['week'] > 127)
   
    assembler = VectorAssembler(
            inputCols = [
                'lag_like_count',
                'lag_skip_count', 
                'lag_playtime_ratio', 
                'lag_1', 
                'lag_2', 
                'lag_3', 
                'lag_4', 
                'lag_5', 
                'popularity', 
                'release_date',
                'duration_ms', 
                'danceability', 
                'acousticness', 
                'instrumentalness'],
        outputCol = "features",
        )

    linear_regression = LinearRegression(
        featuresCol="features",
        labelCol="play_count", 
        predictionCol="predicted_play_count",
    )

    train_data = assembler.transform(train_data)
    test_data = assembler.transform(test_data)

    model = linear_regression.fit(train_data)
    predictions = model.transform(test_data)

    metrics_names = ['rmse', 'mae', 'mse', 'r2']
    metrics_scores = [None for _ in metrics_names]
    metrics = dict(zip(metrics_names, metrics_scores))

    for metric_name in metrics:
        evaluator = RegressionEvaluator(
            labelCol="play_count", 
            predictionCol="predicted_play_count", 
            metricName=metric_name
        )
        
        metric_value = evaluator.evaluate(predictions)
        metrics[metric_name] = metric_value
    
    predictions.write.format("json").save(PREDICTION_DIR, mode="overwrite")
    model.write().overwrite().save(MODEL_DIR)
    
    for metric_name in metrics:
        logger.info(f"{metric_name.upper()}: {metrics[metric_name]}")

    spark.stop()

if __name__ == "__main__":
    main()
