import optuna
import json
import matplotlib.pyplot as plt
from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.ml.regression import GBTRegressor
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.evaluation import RegressionEvaluator

PREDICTION_DIR = "advanced_model/predictions"
MODEL_DIR = "model"
DATA_DIR = "../../data/preprocessed_data.csv"

FEATURES = ['lag_like_count', 'lag_skip_count', 'lag_playtime_ratio',
            'lag_1', 'lag_2', 'lag_3', 'lag_4', 'lag_5', 'popularity', 'release_date',
            'duration_ms', 'danceability', 'acousticness', 'instrumentalness']

TARGET = ['play_count']


def main():
    spark: SparkSession = (SparkSession.builder.appName("GBT Regression Training").getOrCreate())
    logger = spark._jvm.org.apache.log4j.LogManager.getLogger("GBT Regression Training")

    data = spark.read.csv(DATA_DIR, header=True, inferSchema=True)

    train_data = data.filter(data['week'] <= 94)
    validate_data = data.filter(data['week'].isin(95, 125))
    test_data = data.filter(data['week'] > 125)

    assembler = VectorAssembler(
        inputCols=FEATURES,
        outputCol="features",
        )

    train_data = assembler.transform(train_data)
    validate_data = assembler.transform(validate_data)
    test_data = assembler.transform(test_data)

    # hyperparms tuning
    evaluator_tune = RegressionEvaluator(
            labelCol="play_count",
            predictionCol="predicted_play_count",
            metricName='mse'
        )

    def objective(trial):
        # Define GBTRegressor model
        gbt = GBTRegressor(
            featuresCol="features",
            labelCol="play_count",
            predictionCol="predicted_play_count",
            maxDepth=trial.suggest_int("maxDepth", 5, 15),
            maxBins=trial.suggest_int("maxBins", 16, 64),
            maxIter=trial.suggest_int("maxIter", 10, 30)
        )

        # Define pipeline with GBTRegressor
        pipeline = Pipeline(stages=[gbt])

        # Train the model
        model = pipeline.fit(test_data)

        # Validate the model
        predictions = model.transform(validate_data)
        mse = evaluator_tune.evaluate(predictions)

        return mse

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=10)  # Adjust the number of trials based on your needs

    # Get the best hyperparameters
    best_params = study.best_params
    logger.info(f"Best Hyperparameters: {best_params}")
    
    with open('best_params.json', 'w') as json_file:
        json.dump(best_params, json_file)
        
    # Display the optimization results as a DataFrame
    trials_df = study.trials_dataframe()
    print(trials_df)

    # Plot the optimization history
    optuna.visualization.plot_optimization_history(study)
    plt.show()

    # Plot the slice plot of the hyperparameters
    optuna.visualization.plot_slice(study)
    plt.show()

    # train the final model, check metrics
    gbt_regression = GBTRegressor(
            featuresCol="features",
            labelCol="play_count",
            predictionCol="predicted_play_count",
        )

    model = gbt_regression.fit(train_data)
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

    predictions.write.format("json").save(f'{PREDICTION_DIR}', mode="overwrite")
    model.write().overwrite().save(MODEL_DIR)

    for metric_name in metrics:
        logger.info(f"{metric_name.upper()}: {metrics[metric_name]}")

    spark.stop()


if __name__ == "__main__":
    main()
