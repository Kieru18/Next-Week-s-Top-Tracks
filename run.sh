#!/bin/bash
pip install -r requirements.txt
apt-get update
apt-get install openjdk-17-jdk -y
cd models/linear_regression
./train_linear_regression.sh
cd ../gbt_regression
./train_gbt_regression.sh
cd ../../microservice
python3 app.py
