from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer
import pandas as pd
from pyspark.sql.types import StructType, StructField, StringType
# schema = StructType([StructField("A", StringType(), True), StructField("B", StringType(), True)])
schema = StructType(
    [StructField('Solution Key',StringType(),True),
    StructField('PAF Error',StringType(),True),
    StructField('Resolution Steps',StringType(),True),
    StructField('Resolution Strategy',StringType(),True),
    StructField('Need to be monitored',StringType(),True),
    StructField('Source Line Number',StringType(),True),
    StructField('Remote IP Address',StringType(),True),
    StructField('Program/Method/Function Module',StringType(),True),
    StructField('Package',StringType(),True),
    StructField('Name of Method or Function Module',StringType(),True),
    StructField('Name of Class or Program',StringType(),True),
    StructField('Message Area',StringType(),True),
    StructField('Message number',StringType(),True),
    StructField('Expiry Date',StringType(),True),
    StructField('Error Subcategory',StringType(),True),
    StructField('Error Short Text',StringType(),True),
    StructField('Application component ID',StringType(),True),
    StructField('Application Area',StringType(),True),
    StructField('ABAP Name of Consumer or Server Proxy',StringType(),True),
    StructField('Error Log Information',StringType(),True),
    StructField('Sender party',StringType(),True),
    StructField('Sender interface operation',StringType(),True),
    StructField('Sender interface namespace',StringType(),True),
    StructField('Sender interface name',StringType(),True),
    StructField('Receiver interface operation',StringType(),True),
    StructField('Receiver interface namespace',StringType(),True),
    StructField('Receiver interface name',StringType(),True)]
)



spark = SparkSession.builder.getOrCreate()
# df = spark.read.csv("Falcon_HC_Model_Data_CSV.csv",inferSchema=True,header=True)
df=pd.read_excel("Falcon_HC_Model_Data.xlsx",sheet_name=["Model Data"],header=0,index_col=None,na_values=['NA'],usecols="A:AA")
sheetDf = df["Model Data"]
# print(sheetDf.shape)
sdf= spark.createDataFrame(sheetDf,schema=schema)
sdf.show()
# indexers = [StringIndexer(inputCol=column, outputCol=column+"_index") for column in list(set(sdf.columns)) ]
# pipeline = Pipeline(stages=indexers)
# df_r = pipeline.fit(sdf).transform(sdf)
# df_r.show()

# sdf.distinct().show()
