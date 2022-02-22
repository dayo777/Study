# run file
from socket import herror
from sre_parse import SPECIAL_CHARS
from pyspark.sql import SparkSession
from helper import *




# new data to score
path_to_output = '/scores'
filename = 'score_data.csv'
spark = SparkSession.builder.getOrCreate()
score_data = spark.read.csv(filename, header=True, inferSchema=True, sep=';')

# score_new_df is a function from helper.py
final_scores_df = score_new_df(score_data)

final_scores_df.repartition(1).write.format('csv').mode('overwrite').options(sep='|', header='true'). \
    save(path_to_output + '/predictions.csv')


# code was run on Docker


