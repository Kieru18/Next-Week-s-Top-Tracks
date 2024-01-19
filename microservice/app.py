from datetime import datetime
from flask import Flask, request, jsonify
from pyspark.sql import SparkSession, DataFrame
from pyspark.ml.regression import LinearRegressionModel, GBTRegressionModel
from pyspark.ml.feature import VectorAssembler
from pathlib import Path
import json

BASIC_MODEL_DIR = "../models/linear_regression/model"
ADVANCED_MODEL_DIR = "../models/gradient_boosting/model"
DATA_FILE = "../data/preprocessed_data.csv"
DATA_DIR = "../data"
DATA_VER = "v3"

def getWeek(date: str):
    start_date = datetime.strptime('2020-12-28', '%Y-%m-%d')
    date = datetime.strptime(date, '%Y-%m-%d')
    week = (date - start_date).days // 7 + 1
    return week

spark: SparkSession = (SparkSession.builder.appName("microservice").getOrCreate())

app = Flask(__name__)

tracks = None

@app.route("/")
def index():
    return "Hello, World!"

@app.route("/predict", methods=["POST"])
def predict():
    global tracks
    if tracks is None:
        tracks = spark.read.json(f"{DATA_DIR}/{DATA_VER}/tracks.jsonl")

    if request.headers["Content-Type"] != "application/json":
        return jsonify({"error": "Invalid Content-Type"}), 400
    
    data = request.json

    if not {"tracks_num", "date"}.issubset(data.keys()):
        return jsonify({"error": "Invalid JSON"}), 400

    tracks_num = data["tracks_num"]
    date = data["date"]
    week = getWeek(date)
    model_type = data.get("model_type", "advanced")  # Assign "advanced" if not provided

    if (
        model_type not in ["basic", "advanced"] or
        type(date) != str or
        type(tracks_num) != int or
        tracks_num < 1 or
        week < 6
    ):
        return jsonify({"error": "Invalid JSON"}), 400
    
    try:
        toplist = prepare_toplist(tracks_num, week, model_type)
        toplist = toplist.join(tracks.select("name", "id"), toplist.track_id == tracks.id, "left")

        return [json.loads(track) for track in toplist.toJSON().collect()]
    except Exception as e:
        return jsonify({"error": f"Internal server error: {e}"}), 500
    
def prepare_toplist(tracks_num: int, week: int, model_type: str):
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

    data = spark.read.csv(DATA_FILE, header=True, inferSchema=True)
    data = data.filter(data.week == week)

    if model_type == "basic":
        model = LinearRegressionModel.load(BASIC_MODEL_DIR)
        
    elif model_type == "advanced":
        model = GBTRegressionModel.load(ADVANCED_MODEL_DIR)

    df = assembler.transform(data)

    predictions = model.transform(df)
    predictions = predictions.sort("play_count", ascending=False).take(tracks_num)

    if len(predictions) == 0:
        return jsonify({"error": "No predictions"}), 500

    toplist = spark.createDataFrame(predictions)["track_id", "play_count"]
    
    return toplist


def main():
    app.run(host="0.0.0.0", port=5000, debug=True)

if __name__ == "__main__":
    main()