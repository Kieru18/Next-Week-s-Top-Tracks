#!/bin/bash
spark-submit --deploy-mode client --num-executors 5 --executor-memory 2G gbt_regression.py --driver-memory 12G --conf spark.memory.offHeap.enabled=true