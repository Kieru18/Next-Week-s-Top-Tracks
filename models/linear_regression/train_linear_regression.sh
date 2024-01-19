#!/bin/bash
spark-submit --deploy-mode client --num-executors 5 --executor-memory 5G linear_regression.py
