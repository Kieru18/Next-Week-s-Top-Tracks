from pyspark.sql import SparkSession
from pyspark.ml.regression import LinearRegression
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.sql.functions import col
from sklearn.metrics import ndcg_score
import numpy as np
#from pyspark.mllib.evaluation import MulticlassMetrics

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

    train_data = data.filter(data['week'] <= 94)
    test_raw_data = data.filter(data['week'].isin(126, 157))

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
    test_data = assembler.transform(test_raw_data)

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

    ndcg = calc_ndcgAt20(model, test_raw_data, assembler, logger)
    logger.info(f"NDCG@20: {ndcg}")

    for metric_name in metrics:
        logger.info(f"{metric_name.upper()}: {metrics[metric_name]}")

    spark.stop()


def calc_ndcgAt20(model, data, assembler, logger):
    unique_weeks = unique_weeks = data.select("week").distinct().collect()
    unique_weeks = [item for sublist in unique_weeks for item in sublist]
    logger.info(unique_weeks)
    ndcgs = []

    for week in unique_weeks:
        test_week = data.filter(data["week"] == week)
        test_week = assembler.transform(test_week)

        predictions = model.transform(test_week)

        sorted_predictions = predictions.orderBy(col("predicted_play_count").desc())
        ground_truth = sorted_predictions.select("play_count").rdd.flatMap(lambda x: x).collect()
        predicted = sorted_predictions.select("predicted_play_count").rdd.flatMap(lambda x: x).collect()

        ground_truth = np.array(ground_truth).reshape(1, -1)
        predicted = np.array(predicted).reshape(1, -1)

        logger.info(ground_truth)
        logger.info(predicted)

        ndcgs.append(ndcg_score(ground_truth, predicted, k=20))

    ndcg = np.mean(ndcgs)
    return ndcg


if __name__ == "__main__":
    main()
