import fileinput
import pandas as pd  # pip install pandas,pip install openpyxl
import nltk   # pip install nltk
# nltk.download('all')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


receiver=""
sender =""

def errorAnalyzer(error,pafError):
    X_list =error
    # print(pafError)
    if(pafError == "nan" or pafError == ""):
        print("similarity: ", 0)
        return 0

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

def fileAnalyzer(filename,isErrorData):
   
    masterData=""
    for line in fileinput.input(files=filename,encoding="utf-8"):
        if(line.startswith("Receiver interface name:",0)):
            data=line[25:-1]
            masterData+=" "+(data if data!="" else 'nan' )
            if(isErrorData):
                global receiver 
                receiver = data if data!="" else 'nan'
        elif(line.startswith("Receiver interface namespace:",0)):
            data=line[30:-1]
            masterData+=" "+(data if data!="" else 'nan' )
        elif(line.startswith("Receiver interface operation:",0)):
            data=line[30:-1]
            masterData+=" "+(data if data!="" else 'nan' )
        elif(line.startswith("Sender interface name:",0)):
            data=line[23:-1]
            masterData+=" "+(data if data!="" else 'nan' )
            if(isErrorData):
                global sender
                sender=data if data!="" else 'nan'
        elif(line.startswith("Sender interface namespace:",0)):
            data=line[28:-1]
            masterData+=" "+(data if data!="" else 'nan' )
        elif(line.startswith("Sender interface operation:",0)):
            data=line[28:-1]
            masterData+=" "+(data if data!="" else 'nan' )
        elif(line.startswith("Sender party:",0)):
            data=line[14:-1]
            masterData+=" "+(data if data!="" else 'nan' )
        elif(line.startswith("Error Log Information:",0)):
            data=line[23:-1]
            masterData+=" "+(data if data!="" else 'nan' )
        elif(line.startswith("ABAP Name of Consumer or Server Proxy:",0)):
            data=line[40:-1]
            masterData+=" "+(data if data!="" else 'nan' )
        elif(line.startswith("Application Area:",0)):
            data=line[18:-1]
            masterData+=" "+(data if data!="" else 'nan' )
        elif(line.startswith("Application component ID:",0)):
            data=line[26:-1]
            masterData+=" "+(data if data!="" else 'nan' )
        elif(line.startswith("Error Short Text:",0)):
            data=line[18:-1]
            masterData+=" "+(data if data!="" else 'nan' )
        elif(line.startswith("Error Subcategory:",0)):
            data=line[19:-1]
            masterData+=" "+(data if data!="" else 'nan' )
        elif(line.startswith("Expiry Date:",0)):
            data=line[13:-1]
            masterData+=" "+(data if data!="" else 'nan' )
        elif(line.startswith("Message number:",0)):
            data=line[16:-1]
            masterData+=" "+(data if data!="" else 'nan' )
        elif(line.startswith("Message Area:",0)):
            data=line[14:-1]
            masterData+=" "+(data if data!="" else 'nan' )
        elif(line.startswith("Name of Class or Program:",0)):
            data=line[26:-1]
            masterData+=" "+(data if data!="" else 'nan' )
        elif(line.startswith("Name of Method or Function Module:",0)):
            data=line[35:-1]
            masterData+=" "+(data if data!="" else 'nan' )
        elif(line.startswith("Package:",0)):
            data=line[9:-1]
            masterData+=" "+(data if data!="" else 'nan' )
        elif(line.startswith("Program/Method/Function Module:",0)):
            data=line[32:-1]
            masterData+=" "+(data if data!="" else 'nan' )
        elif(line.startswith("Remote IP Address:",0)):
            data=line[19:-1]
            masterData+=" "+(data if data!="" else 'nan' )
        elif(line.startswith("Source Line Number:",0)):
            data=line[20:-1]
            masterData+=" "+(data if data!="" else 'nan' )
    return masterData



errorString=fileAnalyzer('errorInput.txt',True)
X_list = word_tokenize(errorString)
df=pd.read_excel("Falcon_Health_check_Analysis.xlsx",sheet_name=["Solution Bank"],header=0,index_col=None,na_values=['NA'],usecols="A:D,H")


sheetDf = df["Solution Bank"]
# print(sheetDf.columns)
masterString=""
solutions=[]
for index,row in sheetDf.iterrows():
    senderRow= str(row['Sender'])
    recvRow = str(row['Receiver'])
    if(senderRow.strip() == sender and recvRow.strip() == receiver): 
        print(row['Solution Key'])
        lines = str(row['Master Payload'])
        with open('masterPayload.txt', 'w' ,encoding="utf-8") as f:
            f.write(lines)
        masterString =fileAnalyzer('masterPayload.txt',False)
        sim1 = errorAnalyzer(X_list,masterString) * 100
        if(sim1 >=75):
            solutions.append(row["Solution Key"])






if(len(solutions) > 0):
   print("Possible Solution Could Be \n",solutions)
else:
   print("No solution found")




















