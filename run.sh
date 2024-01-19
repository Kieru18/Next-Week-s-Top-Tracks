#!/bin/bash
pip install -r requirements.txt
apt-get update
apt-get install openjdk-17-jdk -y
cd models/linear_regression
spark-submit --deploy-mode client --num-executors 5 --executor-memory 5G linear_regression.py
cd ../../microservice
python3 app.py
