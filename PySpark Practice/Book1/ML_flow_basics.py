from pyspark.sql import SparkSession
import mlflow
import mlflow.spark
import sys
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.sql import functions as F
from pyspark.ml.feature import VectorAssembler
from pyspark.ml import Pipeline


# function to assemble Vectors
def assemble_vectors(df, features_list, target_variable_name):
    stages = []
    assembler = VectorAssembler(inputCols=features_list, outputCol='features')
    stages = [assembler]
    selectedCols = [target_variable_name, 'features']
    pipeline = Pipeline(stages=stages)
    assembleModel = pipeline.fit(df)
    df = assembleModel.transform(df).select(selectedCols)
    return df


spark = SparkSession.builder.appName("mlflow_examples").getOrCreate()
filename = r'bank/bank-full.csv'
target_variable_name = 'y'

df = spark.read.csv(filename, header=True, inferSchema=True, sep=';')
df = df.withColumn('label', F.when(F.col('y') == 'yes', 1).otherwise(0))
df = df.drop('y')

train, test = df.randomSplit([0.7, 0.3], seed=12345)

df = df.select(['age', 'balance', 'day', 'duration', 'campaign', 'pdays', 'previous', 'label'])

# exclude target variable and select all other feature Vectors
features_list = df.columns
features_list.remove('label')

assembled_train_df = assemble_vectors(train, features_list, 'label')
assembled_test_df = assemble_vectors(test, features_list, 'label')

# output the system arguments
print(sys.argv[1])
print(sys.argv[2])


maxBinsVal = float(sys.argv[1]) if len(sys.argv) > 3 else 20
maxDepthVal = float(sys.argv[2]) if len(sys.argv) > 3 else 3



# init Machine-learning with MLFLOW
# this code was run on Docker
with mlflow.start_run():
    stages_tree = []
    classifier = RandomForestClassifier(
        labelCol='label', featuresCol='features', maxBins=maxBinsVal, maxDepth=maxDepthVal
    )
    stages_tree += [classifier]
    pipeline_tree = Pipeline(stages=stages_tree)
    # runnung the RFmodel
    RFmodel = pipeline_tree.fit(assembled_train_df)
    # make predictions
    predictions = RFmodel.transform(assembled_test_df)
    evaluator = BinaryClassificationEvaluator()
    mlflow.log_param("maxBins", maxBinsVal)
    mlflow.log_param("maxDepth", maxDepthVal)
    mlflow.log_metric("ROC", evaluator.evaluate(predictions, {evaluator.metricName: "areaUnderROC"}))
    mlflow.spark.log_model(RFmodel, "spark-model")
