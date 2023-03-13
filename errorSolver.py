import fileinput
import pandas as pd  # pip install pandas,pip install openpyxl
# import re
import nltk   # pip install nltk
import translate


# nltk.download('all')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from translate import Translator
translator = Translator(to_lang="English")

def errorAnalyzer(error,pafError):
    X_list =error
    Y_list = word_tokenize(pafError)
  
    # sw contains the list of stopwords
    sw = stopwords.words('english') 
    l1 =[];l2 =[]
    
    # remove stop words from the string
    X_set = {w for w in X_list if not w in sw} 
    Y_set = {w for w in Y_list if not w in sw}
    
    # form a set containing keywords of both strings 
    rvector = X_set.union(Y_set) 
    for w in rvector:
        if w in X_set: l1.append(1) # create a vector
        else: l1.append(0)
        if w in Y_set: l2.append(1)
        else: l2.append(0)
    c = 0
    
    # cosine formula 
    for i in range(len(rvector)):
            c+= l1[i]*l2[i]
    cosine = c / float((sum(l1)*sum(l2))**0.5)
    print("similarity: ", cosine)
    return cosine

errors=[]
receiver=""
sender =""


for line in fileinput.input(files='errorInput.txt'):
    if(line.startswith("Error information:",0)):
        errors.append(line[19:-1])
    elif(line.startswith("Error Short Text:",0)):
        errors.append(line[18:-1])
    elif(line.startswith("Receiver interface name:",0)):
        receiver=line[25:-1]
    elif(line.startswith("Sender interface name:",0)):
        sender=line[23:-1]
    
receiver = receiver if receiver!="" else 'nan'
sender = sender if sender!="" else 'nan'
# error = translator.translate(errors[0])
# print(errors)
# print(receiver)
# print(sender)
# ","Sender","Solution Keyames=["Receiver" "Sender"]

df=pd.read_excel("Falcon_Health_check_Analysis.xlsx",sheet_name=["Solution Bank"],header=0,index_col=None,na_values=['NA'],usecols="A:D")
# for dfline in df:
#     if(dfline[0] == )
# print(df.head())
# mapping = pd.read_excel(download_path, sheet_name=None)
# for sheet_name in df:
#     sheetDf = df[sheet_name]
#     print(sheetDf.shape)
#     print(sheetDf.columns)
#     # print(sheetDf.head())
    
#     break
sheetDf = df["Solution Bank"]
# print(sheetDf.shape)
# print(sheetDf.columns)
# print(sheetDf.head())
X1_list = word_tokenize(errors[0])
X2_list = word_tokenize(errors[1])
solutions=[]
# print(sheetDf.rows())
for index,row in sheetDf.iterrows():
    # print(row['Sender'])
    senderRow= str(row['Sender'])
    # senderRow=row["Sender"]
    recvRow = str(row['Receiver'])
    # print(senderRow)
    # print(recvRow)
    if(senderRow.strip() == sender and recvRow.strip() == receiver): #  row['PAF Error'] in errors[0]):    # solutions.append(row['Solution Key']);
        print(row['Solution Key'])
        sim1 = errorAnalyzer(X1_list,row["PAF Error"]) * 100
        sim2 = errorAnalyzer(X2_list,row["PAF Error"]) * 100
        if(sim1 >= 25 or sim2 >= 25):
            solutions.append(row["Solution Key"])





if(len(solutions) > 0):
   print("Possible Solution Could Be \n",solutions)
else:
   print("No solution found")




















