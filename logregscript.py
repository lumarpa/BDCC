from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import LogisticRegression
import time
import pandas as pd

def main():
    spark = SparkSession.builder \
        .appName("Parallelized Logistic Regression") \
        .getOrCreate()

    data = spark.read.csv("s3://mylogreg/data_input/flights.csv", header=True, inferSchema=True)

    feature_columns = [col for col in data.columns if col != ('delayed' and "ArrDelay" and "Origin")]
    assembler = VectorAssembler(inputCols=feature_columns, outputCol="features")
    data_prepared = assembler.transform(data)
    start_time = time.time()
    lr = LogisticRegression(featuresCol="features", labelCol="delayed")
    lr_model = lr.fit(data_prepared)

    spark.stop()
    return start_time

if __name__ == "__main__":
    start_time = main()
    end_time = time.time()
    print("Execution time: {} seconds".format(end_time - start_time))
    df = pd.DataFrame({"time": [end_time-start_time]})
    df.to_csv("time.csv")