from pyspark.ml import Pipeline
from pyspark.ml.classification import GBTClassifier
from pyspark.ml.feature import StringIndexer, VectorIndexer, VectorAssembler
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

# Load and parse the data file, converting it to a DataFrame.
data = spark.read.format("csv").option("header", "true").load("data/gdelt_encoded_full.csv")
stages = []
# Index labels, adding metadata to the label column.
# Fit on whole dataset to include all labels in index.

# Automatically identify categorical features, and index them.
# Set maxCategories so features with > 4 distinct values are treated as continuous.
numericCols = ['Source','Target','NumEvents','NumArts','SourceGeoType',
      'TargetGeoType', 'ActionGeoType','Month']
assemblerInputs = numericCols
data.printSchema()
assembler = VectorAssembler(inputCols=assemblerInputs, outputCol="features")
stages += [assembler]
pipeline = Pipeline(stages = stages)
pipelineModel = pipeline.fit(data)
df = pipelineModel.transform(data)
selectedCols = ['label', 'features'] + data.columns
df = df.select(selectedCols)

