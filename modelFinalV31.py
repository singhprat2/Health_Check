from pyspark.sql import SparkSession
from pyspark.ml.feature import StringIndexer
import pandas as pd
from pyspark.sql.functions import col
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import NaiveBayes
from pyspark.ml.pipeline import PipelineModel
import fileinput
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import os
from flask import Flask,jsonify,request
from flask_cors import CORS, cross_origin
app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'
port = int(os.environ.get('PORT', 2000))
@cross_origin()

@app.route('/load_Df_train', methods=['GET','POST'])
def load_Df_train(spark):
    return spark.read.json("./df_train")

def load_indexed_train(spark):
    return spark.read.json("./indexed_train")

def loadModel():
    mPath =  "./model"
    return PipelineModel.load(mPath)

def tokennizeData(data):
    sw = stopwords.words('english')
    # remove stop words from the string
    X_list = word_tokenize(data)
    X_set = {w for w in X_list if not w in sw}
    return X_set

# To get data from master Payload

def fileAnalyzer(filename,masterDict):
    try:
        nltk.download('stopwords')
    except Exception as e:
            return("there is error with nltk downloader")
    errorInformationVar=""
    try:
        for line in fileinput.input(files=filename, encoding="utf-8"):
            if(line.startswith("Receiver interface name:", 0)):
                data = line[25:-1]

                data = data.strip()
                data = data.strip('" \"')

                masterDict['Receiver interface name'] = (
                    data if data != "" else 'nan')
                # if(isErrorData):
                #     global receiver
                #     receiver = data if data!="" else 'nan'
            elif(line.startswith("Receiver interface namespace:", 0)):
                data = line[30:-1]
                data = data.strip()
                data = data.strip('" \"')

                masterDict['Receiver interface namespace'] = (
                    data if data != "" else 'nan')
            elif(line.startswith("Receiver interface operation:", 0)):
                data = line[30:-1]
                data = data.strip()
                data = data.strip('" \"')

                masterDict['Receiver interface operation'] = (
                    data if data != "" else 'nan')
            elif(line.startswith("Sender interface name:", 0)):
                data = line[23:-1]
                data = data.strip()
                data = data.strip('" \"')

                masterDict['Sender interface name'] = (
                    data if data != "" else 'nan')
                # if(isErrorData):
                #     global sender
                #     sender=data if data!="" else 'nan'
            elif(line.startswith("Sender interface namespace:", 0)):
                data = line[28:-1]
                data = data.strip()
                data = data.strip('" \"')

                masterDict['Sender interface namespace'] = (
                    data if data != "" else 'nan')
            elif(line.startswith("Sender interface operation:", 0)):
                data = line[28:-1]
                data = data.strip()
                data = data.strip('" \"')

                masterDict['Sender interface operation'] = (
                    data if data != "" else 'nan')
            elif(line.startswith("Sender party:", 0)):
                data = line[14:-1]
                data = data.strip()
                data = data.strip('" \"')

                masterDict['Sender party'] = (data if data != "" else 'nan')
            elif(line.startswith("Error Log Information:", 0)):
                data = line[23:-1]
                data = data.strip()
                data = data.strip('" \"')

                data = tokennizeData(data)
                data=sorted(data,key=str.casefold)
                masterDict['Error Log Information'] = (
                    data if data != "" else 'nan')
            elif(line.startswith("Error information:", 0)):
                
                data = line[19:-1]
                errorInformationVar=data
                data = data.strip()
                data = data.strip('" \"')

                data = tokennizeData(data)
                data=sorted(data,key=str.casefold)
                masterDict['Error Information'] = (
                    data if data != "" else 'nan')
            elif(line.startswith("ABAP Name of Consumer or Server Proxy:", 0)):
                data = line[40:-1]
                data = data.strip()
                data = data.strip('" \"')


                masterDict['ABAP Name of Consumer or Server Proxy'] = (
                    data if data != "" else 'nan')
            elif(line.startswith("Application Area:", 0)):
                data = line[18:-1]
                data = data.strip()
                data = data.strip('" \"')


                masterDict['Application Area'] = (data if data != "" else 'nan')
            elif(line.startswith("Application component ID:", 0)):
                data = line[26:-1]
                data = data.strip()
                data = data.strip('" \"')

                masterDict['Application component ID'] = (
                    data if data != "" else 'nan')
            elif(line.startswith("Error Short Text:", 0)):
                data = line[18:-1]
                data = data.strip()
                data = data.strip('" \"')

                data = tokennizeData(data)
                data=sorted(data,key=str.casefold)
                masterDict['Error Short Text'] = (data if data != "" else 'nan')
                # masterDict['PAF Error']=(data if data != "" else 'nan' )
            elif(line.startswith("Error Subcategory:", 0)):
                data = line[19:-1]
                data = data.strip()
                data = data.strip('" \"')

                data = tokennizeData(data)
                data=sorted(data,key=str.casefold)
                masterDict['Error Subcategory'] = (
                    data if data != "" else 'nan')
            elif(line.startswith("Expiry Date:", 0)):
                data = line[13:-1]
                data = data.strip()
                data = data.strip('" \"')

                masterDict['Expiry Date'] = (data if data != "" else 'nan')
            elif(line.startswith("Message number:", 0)):
                data = str(line[16:-1])
                data = data.strip()
                data = data.strip('" \"')

                masterDict['Message number'] = (data if data != "" else 'nan')
            elif(line.startswith("Message Area:", 0)):
                data = line[14:-1]
                data = data.strip()
                data = data.strip('" \"')

                masterDict['Message Area'] = (data if data != "" else 'nan')
            elif(line.startswith("Name of Class or Program:", 0)):
                data = line[26:-1]
                data = data.strip()
                data = data.strip('" \"')

                masterDict['Name of Class or Program'] = (
                    data if data != "" else 'nan')
            elif(line.startswith("Name of Method or Function Module:", 0)):
                data = line[35:-1]
                data = data.strip()
                data = data.strip('" \"')

                masterDict['Name of Method or Function Module'] = (
                    data if data != "" else 'nan')
            elif(line.startswith("Package:", 0)):
                data = line[9:-1]
                data = data.strip()
                data = data.strip('" \"')

                masterDict['Package'] = (data if data != "" else 'nan')
            elif(line.startswith("Program/Method/Function Module:", 0)):
                data = line[32:-1]
                data = data.strip()
                data = data.strip('" \"')
                # print(data)
                masterDict['Program/Method/Function Module'] = (
                    data if data != "" else 'nan')
            elif(line.startswith("Remote IP Address:", 0)):
                data = line[19:-1]
                data = data.strip()
                data = data.strip('" \"')

                masterDict['Remote IP Address'] = (
                    data if data != "" else 'nan')
            elif(line.startswith("Source Line Number:", 0)):
            # print(line)
                data = (line[20:])
                # print(data)
                data = data.strip()
                data = data.strip('" \"')

            masterDict['Source Line Number'] = (
                data if data != "" else 'nan')
        fileinput.close()
    except Exception as e:
            return("there is error with file reading")
    return errorInformationVar


# Processing The master Payload
@app.route('/inputPayload', methods=['GET','POST'])
def inputPayload():
    masterPayload = request.get_json()
    # if(masterPayload !=""):
    #     return jsonify(masterPayload)
    masterPayload=masterPayload if masterPayload !="" else ""
    result={
        'sender':"",
        'recv':"",
        'Error':"",
        'ErrorId':"",
        'UniqueKeys':[],
        'PercentageMatch':[],
        'ResolutionsSteps':[]
    }
    masterDict={
    'Solution Key':'nan',
    'Error Information':'nan',
    'Source Line Number':'nan',
    'Remote IP Address':'nan',
    'Program/Method/Function Module':'nan',
    'Package':'nan',
    'Name of Method or Function Module':'nan',
    'Name of Class or Program':'nan',
    'Message Area':'nan',
    'Message number':'nan',
    'Expiry Date':'nan',
    'Error Subcategory':'nan',
    'Error Short Text':'nan',
    'Application component ID':'nan',
    'Application Area':'nan',
    'ABAP Name of Consumer or Server Proxy':'nan',
    'Error Log Information':'nan',
    'Sender party':'nan',
    'Sender interface operation':'nan',
    'Sender interface namespace':'nan',
    'Sender interface name':'nan',
    'Receiver interface operation':'nan',
    'Receiver interface namespace':'nan',
    'Receiver interface name':'nan'
    }
    return jsonify(masterPayload)
    if(masterPayload != ""):
        try:
            with open('errorInput.txt', 'w' ,encoding="utf-8") as f:
                f.write(masterPayload)
            f.close()
        except Exception as e:
            return("there is error with file writing")

    try:
        result['Error']= fileAnalyzer('errorInput.txt',masterDict)
    except Exception as e:
            return("there is error with fileAnalyzer method")
    
    
    result['sender']=masterDict['Sender interface name']
    result['recv']=masterDict['Receiver interface name']
    # result['Error']=masterDict['Error Information']
    result['ErrorId']=masterDict['Message Area']+":"+masterDict['Message number']
    df = pd.DataFrame.from_dict(masterDict, orient='index')
    df = df.transpose()
    # Converting Data to CSV
    df.to_csv('my_file.csv', index=False, header=True)

    result['UniqueKeys'].append(getPredictionForTest())
    return jsonify(result)



def top3_index(lst, rev=True):
    index = range(len(lst))
    s = sorted(index, reverse=rev, key=lambda i: lst[i])
    return s[:3]

def load_indexer():
    return StringIndexer(inputCols=[
    'Solution Key',
    'Error Information',
    'Source Line Number',
    'Remote IP Address',
    'Program/Method/Function Module',
    'Package',
    'Name of Method or Function Module',
    'Name of Class or Program',
    'Message Area',
    'Message number',
    'Expiry Date',
    'Error Subcategory',
    'Error Short Text',
    'Application component ID',
    'Application Area',
    'ABAP Name of Consumer or Server Proxy',
    'Error Log Information',
    'Sender party',
    'Sender interface operation',
    'Sender interface namespace',
    'Sender interface name',
    'Receiver interface operation',
    'Receiver interface namespace',
    'Receiver interface name'
],
                        outputCols=[
    'label',
    'Error Information_index',
    'Source Line Number_index',
    'Remote IP Address_index',
    'Program/Method/Function Module_index',
    'Package_index',
    'Name of Method or Function Module_index',
    'Name of Class or Program_index',
    'Message Area_index',
    'Message number_index',
    'Expiry Date_index',
    'Error Subcategory_index',
    'Error Short Text_index',
    'Application component ID_index',
    'Application Area_index',
    'ABAP Name of Consumer or Server Proxy_index',
    'Error Log Information_index',
    'Sender party_index',
    'Sender interface operation_index',
    'Sender interface namespace_index',
    'Sender interface name_index',
    'Receiver interface operation_index',
    'Receiver interface namespace_index',
    'Receiver interface name_index']).setHandleInvalid("keep")

def vectorAssembler(indexed_to_pred):
    numericCols = [
    'Error Information_index',
    'Source Line Number_index',
    'Remote IP Address_index',
    'Program/Method/Function Module_index',
    'Package_index',
    'Name of Method or Function Module_index',
    'Name of Class or Program_index',
    'Message Area_index',
    'Message number_index',
    'Expiry Date_index',
    'Error Subcategory_index',
    'Error Short Text_index',
    'Application component ID_index',
    'Application Area_index',
    'ABAP Name of Consumer or Server Proxy_index',
    'Error Log Information_index',
    'Sender party_index',
    'Sender interface operation_index',
    'Sender interface namespace_index',
    'Sender interface name_index',
    'Receiver interface operation_index',
    'Receiver interface namespace_index',
    'Receiver interface name_index'
    ]
    assembler_to_pred = VectorAssembler(inputCols=numericCols, outputCol="features")
    return assembler_to_pred.transform(indexed_to_pred)


def getPredictionForTest():
    spark = SparkSession.builder.getOrCreate()
    print(spark.version)
    df_to_pred = spark.read.csv("my_file.csv",inferSchema=False,header=True)
    # indexed_to_pred = indexer.fit(df_train).transform(df_to_pred)
    indexer =load_indexer()
    df_train = load_Df_train(spark)
    indexed_train = load_indexed_train(spark)

    indexed_to_pred = indexer.fit(df_train).transform(df_to_pred)
    indexed_to_pred = vectorAssembler(indexed_to_pred)
    mPath =  "model"

    from pyspark.ml.pipeline import PipelineModel
    persistedModel = PipelineModel.load(mPath)
   
    nb_predictions=persistedModel.transform(indexed_to_pred)
    nb_prob_list=nb_predictions.collect()[0][50]
    top3_Lebel_res=top3_index(nb_prob_list)
    top3_res=[]
    for x in top3_Lebel_res:
        percentage=(round(nb_prob_list[x],2)*100)
        uniq_key=(indexed_train.filter(indexed_train.label==x).collect()[0][1])
        top3_res.append((percentage,uniq_key))
    print(top3_res)
    return top3_res


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=port)
