from pyspark.ml import Pipeline
from pyspark.ml.classification import GBTClassifier
from pyspark.ml.feature import StringIndexer, VectorIndexer, VectorAssembler
import org.apache.spark.ml.classification.{RandomForestClassificationModel, RandomForestClassifier}
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.sql import functions as F
# Load and parse the data file, converting it to a DataFrame.
data = spark.read.format("csv").option("header", "true").load("data/gdelt_encoded_full.csv")
stages = []
for col in data.columns:
  data = data.withColumn(
    col,
    F.col(col).cast("double")
  )
# Index labels, adding metadata to the label column.
# Fit on whole dataset to include all labels in index.
rf = RandomForestClassifier(labelCol="CAMEOCode", featuresCol="features", numTrees=10)
# Automatically identify categorical features, and index them.
# Set maxCategories so features with > 4 distinct values are treated as continuous.
numericCols = ['Source','Target','NumEvents','NumArts','SourceGeoType',
      'TargetGeoType', 'ActionGeoType','Month']
assemblerInputs = numericCols
data.printSchema()
assembler = VectorAssembler(inputCols=assemblerInputs, outputCol="features")
stages += [assembler]
stages += [rf]
pipeline = Pipeline(stages = stages)
(trainingData, testData) = data.randomSplit([0.7, 0.3])
pipelineModel = pipeline.fit(trainingData)

predictions = pipelineModel.transform(testData)
predictions.select("prediction", "CAMEOCode", "features").show(5)

# Select (prediction, true label) and compute test error
evaluator = MulticlassClassificationEvaluator(
    labelCol="CAMEOCode", predictionCol="prediction", metricName="accuracy")
accuracy = evaluator.evaluate(predictions)
print("Test Error = %g" % (1.0 - accuracy))

