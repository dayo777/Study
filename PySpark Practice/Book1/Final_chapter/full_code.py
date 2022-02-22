import os, pickle
from pyspark.sql import SparkSession
from pyspark.ml.classification import RandomForestClassifier
from pyspark.sql import functions as F
from pyspark.mllib.evaluation import MulticlassMetrics, BinaryClassificationMetrics
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.sql.types import DoubleType
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml import Pipeline
import seaborn as sns
import matplotlib.pyplot as plt

filename = r'bank/bank-full.csv'
target_varaible = 'y'

# read the file
spark = SparkSession.builder.getOrCreate()
df = spark.read.csv(filename, header=True, inferSchema=True, sep=';')

# convert target column to numeric
df = df.withColumn('y', F.when(F.col('y') == 'yes', 1).otherwise(0))

# function to identify DataTypes, ...dtypes outputs tuples of (column_name, column_dtype)
def variable_type(df):
    vars_list = df.types
    char_vars, num_vars = [], []
    for i in vars_list:
        if i[1] in ('string'):
            char_vars.append(i[0])
        else:
            num_vars.append(i[0])

char_vars, num_vars = variable_type(df)
num_vars.remove(target_varaible)

# convert categorical to numeric using Label Encoder
def category_to_index(df, char_vars):
    char_df = df.select(char_vars)
    indexers = [StringIndexer(
        inputCol=c, outputCol=c+'_index', handleInvalid='keep'
    ) for c in char_df.columns]
    pipeline = Pipeline(stages=indexers)
    char_labels = pipeline.fit(char_df)
    df = char_labels.transform(df)
    return df, char_labels

df, char_labels = category_to_index(df, char_vars)
df = df.select([c for c in df.columns if c not in char_vars])

# rename encoded columns to original variable name
def rename_columns(df, char_vars):
    mapping = dict(zip([i + '_index' for i in char_vars], char_vars))
    df = df.select([F.col(c).alias(mapping.get(c, c)) for c in df.columns])
    return df

df = rename_columns(df, char_vars)

# assemble individual columns to one column = 'features'
def assemble_vectors(df, features_list, target_variable):
    stages = []
    assembler = VectorAssembler(inputCols=features_list, outputCol='features')
    stages = [assembler]
    selectedCols = [target_varaible,'features'] + features_list
    pipeline = Pipeline(stages=stages)
    assembleModel = pipeline.fit(df)
    df = assembleModel.transform(df).select(selectedCols)
    return df, assembleModel, selectedCols


# exclude target variable & select all other feature vector
features_list = df.columns
features_list.remove(target_varaible)

# apply the function on our DataFrame
df, assembleModel, selectedCols = assemble_vectors(df, features_list, target_varaible)
train, test = df.randomSplit([0.7, 0.3], seed=12345)

# initialize RandomForestClassifier
clf = RandomForestClassifier(featuresCol='features', labelCol='y')
clf_model = clf.fit(train)
train_pred_result = clf_model.transform(train)
test_pred_result = clf_model.transform(test)

# validate Random forest model
def evaluation_metrics(df, target_variable):
    pred = df.select('prediction', target_variable)
    pred = pred.withColumn(target_varaible, pred[target_varaible].cast(DoubleType()))
    pred = pred.withColumn('prediction', pred['prediction'].cast(DoubleType()))
    metrics = MulticlassMetrics(pred.rdd.map(tuple))
    cm = metrics.confusionMatrix().toArray()
    acc = metrics.accuracy
    misclassification_rate = 1 - acc
    precision = metrics.precision(1.0) # precision
    recall = metrics.recall(1.0) # recall
    f1 = metrics.fMeasure(1.0) # f1 score
    evaluator_roc = BinaryClassificationEvaluator(
        labelCol=target_varaible, rawPredictionCol='rawPrediction', metricName='areaUnderROC'
    )
    roc = evaluator_roc.evaluate(df)
    evaluator_pr = BinaryClassificationEvaluator(
        labelCol=target_varaible, rawPredictionCol='rawPrediction', metricName='areaUnderPR'
    )
    pr = evaluator_pr.evaluate(df)
    return cm, acc, misclassification_rate, precision, recall, f1, roc, pr

train_cm, train_acc, train_miss_rate, train_precision, train_recall, \
    train_f1, train_roc, train_pr = evaluation_metrics(train_pred_result, target_varaible)

test_cm, test_acc, test_miss_rate, test_precision, test_recall, test_f1, \
    test_roc, test_pr = evaluation_metrics(test_pred_result, target_varaible)


def make_confusion_matrix_chart(cf_matrix_train, cf_matrix_test):
    list_values = ['0', '1']
    plt.figure(1, figsize=(10, 5))
    plt.subplot(121)
    sns.heatmap(
        cf_matrix_train, annot=True, yticklabels=list_values, xticklabels=list_values, fmt='g'
    )
    plt.ylabel('Actual')
    plt.xlabel('Pred')
    plt.ylim([0, len(list_values)])
    plt.title('Test data predictions')
    plt.tight_layout()
    return None

make_confusion_matrix_chart(train_cm, test_cm)

# make ROC chart & PR curve
class CurveMetrics(BinaryClassificationMetrics):
    def __init__(self, *args):
        super(CurveMetrics, self).__init__(*args)
    
    def _to_list(self, rdd):
        points = []
        results_collect = rdd.collect()
        for row in results_collect:
            points += [(float(row._1()), float(row._2()))]
        return points
    
    def get_curve(self, method):
        rdd = getattr(self._java_model, method)().toJavaRDD()
        return self._to_list(rdd)
    # end of class

def plot_roc_pr(df, target_variable, plot_type, legend_value, title):
    preds = df.select(target_varaible, 'probability')
    preds = preds.rdd.map(
        lambda row: (float(row['probability'][1]), float(row[target_varaible]))
    )
    points = CurveMetrics(preds).get_curve(plot_type)
    plt.figure()
    x_val = [x[0] for x in points]
    y_val = [x[1] for x in points]
    plt.title(title)

    if plot_type == 'roc':
        plt.xlabel('False Positive Rate (1-Specificity)')
        plt.ylabel('True Positive Rate (Sensitivity)')
        plt.plot(x_val, y_val, label = 'AUC = %0.2f' % legend_value)
        plt.plot([0, 1], [0, 1], color='red', linestyle='--')

    if plot_type == 'pr':
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.plot(x_val, y_val, label = 'Average Precision = %0.2f' % legend_value)
        plt.plot([0, 1], [0.5, 0.5], color='red', linestyle='--')

    plt.legend(loc='lower right')
    return None

plot_roc_pr(train_pred_result, target_varaible, 'roc', train_roc, 'TRAIN_ROC')
plot_roc_pr(test_pred_result, target_varaible, 'roc', test_roc, 'TEST_ROC')
plot_roc_pr(train_pred_result, target_varaible, 'pr', train_pr, 'TrainPrecision-Recall curve')
plot_roc_pr(test_pred_result, target_varaible, 'pr', test_pr, 'TestPrecision-Recall curve')


#-------------------------------------------------------------------
# The model objects we need for Reusability
'''
    char_labels -> Label encoding on new data,
    assembleModel -> assemble vectors and ready data for scoring,
    clf_model -> the RandomForestClassifier,
    features_list -> list of input features to be used for the data,
    char_vars -> character variables,
    num_vars -> numeric variables,
'''



output_path = r'\path_outputs'
# check if DIR exists
try:
    os.mkdir(output_path)
except:
    pass

# save pyspark objects
char_labels.write().overwrite().save(output_path + '/char_label_model.h5')
assembleModel.write().overwrite().save(output_path + '/assembleModel.h5')
clf_model.write().overwrite().save(output_path + '/clf_model.h5')

# save Python object
list_of_vars = [features_list, char_vars, num_vars]
with open(output_path + '/file.pkl', 'wb') as handle:
    pickle.dump(list_of_vars, handle)


print('DONE!!!!')