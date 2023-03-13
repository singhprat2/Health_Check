import fileinput
import pandas as pd
from openpyxl import load_workbook  # pip install pandas,pip install openpyxl
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
masterDict={
    'Solution Key':[],
    # 'PAF Error':[],
    # 'Resolution Steps':[],
    # 'Resolution Strategy':[],
    # 'Need to be monitored':[],
    'Error Information':[],
    'Source Line Number':[],
    'Remote IP Address':[],
    'Program/Method/Function Module':[],
    'Package':[],
    'Name of Method or Function Module':[],
    'Name of Class or Program':[],
    'Message Area':[],
    'Message number':[],
    'Expiry Date':[],
    'Error Subcategory':[],
    'Error Short Text':[],
    'Application component ID':[],
    'Application Area':[],
    'ABAP Name of Consumer or Server Proxy':[],
    'Error Log Information':[],
    'Sender party':[],
    'Sender interface operation':[],
    'Sender interface namespace':[],
    'Sender interface name':[],
    'Receiver interface operation':[],
    'Receiver interface namespace':[],
    'Receiver interface name':[]
}
# def checkdata(tokenString):
#     {

#     }
   
# masterDict={
    # 'Solution Key_index'
    # 'PAF Error_index'
    # 'Resolution Steps_index'
    # 'Resolution Strategy_index'
    # 'Need to be monitored_index'
    # 'Source Line Number_index'
    # 'Remote IP Address_index'
    # 'Program/Method/Function Module_index'
    # 'Package_index'
    # 'Name of Method or Function Module_index'
    # 'Name of Class or Program_index'
    # 'Message Area_index'
    # 'Message number_index'
    # 'Expiry Date_index'
    # 'Error Subcategory_index'
    # 'Error Short Text_index'
    # 'Application component ID_index'
    # 'Application Area_index'
    # 'ABAP Name of Consumer or Server Proxy_index'
    # 'Error Log Information_index'
    # 'Sender party_index'
    # 'Sender interface operation_index'
    # 'Sender interface namespace_index'
    # 'Sender interface name_index'
    # 'Receiver interface operation_index'
    # 'Receiver interface namespace_index'
    # 'Receiver interface name_index'
# }
def tokennizeData(data):
    sw = stopwords.words('english')
    # remove stop words from the string
    X_list = word_tokenize(data)
    X_set = {w for w in X_list if not w in sw}
    return X_set


def handelNull():
    global masterDict
    data = ""
    masterDict['Error Information'].append(data if data != "" else 'nan')
    masterDict['Receiver interface name'].append(data if data != "" else 'nan')
    masterDict['Receiver interface namespace'].append(data if data != "" else 'nan')
    masterDict['Receiver interface operation'].append(data if data != "" else 'nan')
    masterDict['Sender interface name'].append(data if data != "" else 'nan')
    masterDict['Sender interface namespace'].append(data if data != "" else 'nan')
    masterDict['Sender interface operation'].append(data if data != "" else 'nan')
    masterDict['Sender party'].append(data if data != "" else 'nan')
    masterDict['Error Log Information'].append(data if data != "" else 'nan')
    masterDict['ABAP Name of Consumer or Server Proxy'].append(data if data != "" else 'nan')
    masterDict['Application Area'].append(data if data != "" else 'nan')
    masterDict['Application component ID'].append(data if data != "" else 'nan')
    masterDict['Error Short Text'].append(data if data != "" else 'nan')
    masterDict['Error Subcategory'].append(data if data != "" else 'nan')
    masterDict['Expiry Date'].append(data if data != "" else 'nan')
    masterDict['Message number'].append(data if data != "" else 'nan')
    masterDict['Message Area'].append(data if data != "" else 'nan')
    masterDict['Name of Class or Program'].append(data if data != "" else 'nan')
    masterDict['Name of Method or Function Module'].append(data if data != "" else 'nan')
    masterDict['Package'].append(data if data != "" else 'nan')
    masterDict['Program/Method/Function Module'].append(data if data != "" else 'nan')
    masterDict['Remote IP Address'].append(data if data != "" else 'nan')
    masterDict['Source Line Number'].append(data if data != "" else 'nan')
    masterDict['Solution Key'].append(data if data != "" else 'nan')
    # masterDict['PAF Error'].append(data if data != "" else 'nan')
    # masterDict['Resolution Steps'].append(data if data != "" else 'nan')
    # masterDict['Need to be monitored'].append(data if data != "" else 'nan')
    # masterDict['Resolution Strategy'].append(data if data != "" else 'nan')


def fileAnalyzer(filename, i):

    global masterDict
    for line in fileinput.input(files=filename, encoding="utf-8"):
        if(line.startswith("Receiver interface name:", 0)):
            data = line[25:-1]

            data = data.strip()
            data = data.strip('" \"')

            masterDict['Receiver interface name'][i] = (
                data if data != "" else 'nan')
            # if(isErrorData):
            #     global receiver
            #     receiver = data if data!="" else 'nan'
        elif(line.startswith("Receiver interface namespace:", 0)):
            data = line[30:-1]
            data = data.strip()
            data = data.strip('" \"')

            masterDict['Receiver interface namespace'][i] = (
                data if data != "" else 'nan')
        elif(line.startswith("Receiver interface operation:", 0)):
            data = line[30:-1]
            data = data.strip()
            data = data.strip('" \"')

            masterDict['Receiver interface operation'][i] = (
                data if data != "" else 'nan')
        elif(line.startswith("Sender interface name:", 0)):
            data = line[23:-1]
            data = data.strip()
            data = data.strip('" \"')

            masterDict['Sender interface name'][i] = (
                data if data != "" else 'nan')
            # if(isErrorData):
            #     global sender
            #     sender=data if data!="" else 'nan'
        elif(line.startswith("Sender interface namespace:", 0)):
            data = line[28:-1]
            data = data.strip()
            data = data.strip('" \"')

            masterDict['Sender interface namespace'][i] = (
                data if data != "" else 'nan')
        elif(line.startswith("Sender interface operation:", 0)):
            data = line[28:-1]
            data = data.strip()
            data = data.strip('" \"')

            masterDict['Sender interface operation'][i] = (
                data if data != "" else 'nan')
        elif(line.startswith("Sender party:", 0)):
            data = line[14:-1]
            data = data.strip()
            data = data.strip('" \"')

            masterDict['Sender party'][i] = (data if data != "" else 'nan') 
        elif(line.startswith("Error information:", 0)):
            data = line[19:-1]
            data = data.strip()
            data = data.strip('" \"')

            data = tokennizeData(data)
            data=sorted(data,key=str.casefold)
            masterDict['Error Information'][i] = (
                data if data != "" else 'nan')            
        elif(line.startswith("Error Log Information:", 0)):
            data = line[23:-1]
            data = data.strip()
            data = data.strip('" \"')

            data = tokennizeData(data)
            data=sorted(data,key=str.casefold)
            masterDict['Error Log Information'][i] = (
                data if data != "" else 'nan')
        elif(line.startswith("ABAP Name of Consumer or Server Proxy:", 0)):
            data = line[40:-1]
            data = data.strip()
            data = data.strip('" \"')


            masterDict['ABAP Name of Consumer or Server Proxy'][i] = (
                data if data != "" else 'nan')
        elif(line.startswith("Application Area:", 0)):
            data = line[18:-1]
            data = data.strip()
            data = data.strip('" \"')


            masterDict['Application Area'][i] = (data if data != "" else 'nan')
        elif(line.startswith("Application component ID:", 0)):
            data = line[26:-1]
            data = data.strip()
            data = data.strip('" \"')

            masterDict['Application component ID'][i] = (
                data if data != "" else 'nan')
        elif(line.startswith("Error Short Text:", 0)):
            data = line[18:-1]
            data = data.strip()
            data = data.strip('" \"')

            data = tokennizeData(data)
            data=sorted(data,key=str.casefold)
            masterDict['Error Short Text'][i] = (data if data != "" else 'nan')
        elif(line.startswith("Error Subcategory:", 0)):
            data = line[19:-1]
            data = data.strip()
            data = data.strip('" \"')

            data = tokennizeData(data)
            data=sorted(data,key=str.casefold)
            masterDict['Error Subcategory'][i] = (
                data if data != "" else 'nan')
        elif(line.startswith("Expiry Date:", 0)):
            data = line[13:-1]
            data = data.strip()
            data = data.strip('" \"')

            masterDict['Expiry Date'][i] = (data if data != "" else 'nan')
        elif(line.startswith("Message number:", 0)):
            data = line[16:-1]
            data = data.strip()
            data = data.strip('" \"')

            masterDict['Message number'][i] = (data if data != "" else 'nan')
        elif(line.startswith("Message Area:", 0)):
            data = line[14:-1]
            data = data.strip()
            data = data.strip('" \"')

            masterDict['Message Area'][i] = (data if data != "" else 'nan')
        elif(line.startswith("Name of Class or Program:", 0)):
            data = line[26:-1]
            data = data.strip()
            data = data.strip('" \"')

            masterDict['Name of Class or Program'][i] = (
                data if data != "" else 'nan')
        elif(line.startswith("Name of Method or Function Module:", 0)):
            data = line[35:-1]
            data = data.strip()
            data = data.strip('" \"')

            masterDict['Name of Method or Function Module'][i] = (
                data if data != "" else 'nan')
        elif(line.startswith("Package:", 0)):
            data = line[9:-1]
            data = data.strip()
            data = data.strip('" \"')

            masterDict['Package'][i] = (data if data != "" else 'nan')
        elif(line.startswith("Program/Method/Function Module:", 0)):
            data = line[32:-1]
            data = data.strip()
            data = data.strip('" \"')
            # print(data)
            masterDict['Program/Method/Function Module'][i] = (
                data if data != "" else 'nan')
        elif(line.startswith("Remote IP Address:", 0)):
            data = line[19:-1]
            data = data.strip()
            data = data.strip('" \"')

            masterDict['Remote IP Address'][i] = (
                data if data != "" else 'nan')
        elif(line.startswith("Source Line Number:", 0)):
            # print(line)
            data = (line[20:])
            # print(data)
            data = data.strip()
            data = data.strip('" \"')

            masterDict['Source Line Number'][i] = (
                data if data != "" else 'nan')

    # return masterDict
df = pd.read_excel("Falcon_Health_check_Analysis.xlsx", sheet_name=[
                   "Solution Bank"], header=0, index_col=None, na_values=['NA'], usecols="A:H")

df1 = pd.read_excel("Falcon_Health_check_Analysis.xlsx", sheet_name=[
                    "2023 Incidents Tracker"], header=0, index_col=None, na_values=['NA'], usecols="B:J")


sheetDf = df["Solution Bank"]
sheetDf1 = df1["2023 Incidents Tracker"]
i = 0
for index, row in sheetDf.iterrows():
    handelNull()
    data = str(row['Solution Key'])
    data = data.strip()    
    masterDict['Solution Key'][i] = (data if data != "" else 'nan')
    # data = str(row['PAF Error'])
    # data = data.strip()
    # data = data.strip('" \"')
    # data = tokennizeData(data)
    # data=sorted(data,key=str.casefold)
    # masterDict['PAF Error'][i] = (data if data != "" else 'nan')
    # data = str(row['Resolution Steps'])
    # data = data.strip()
    # data = data.strip('" \"')

    # data = tokennizeData(data)
    data=sorted(data,key=str.casefold)
    # masterDict['Resolution Steps'][i] = (data if data != "" else 'nan')
    # data = str(row['Resolution Strategy'])
    # data = data.strip()
    # data = data.strip('" \"')

    # data = tokennizeData(data)
    data=sorted(data,key=str.casefold)
    # masterDict['Resolution Strategy'][i] = (data if data != "" else 'nan')
    # data = str(row['Need to be monitored'])
    # data = data.strip()
    # data = data.strip('" \"')

    # masterDict['Need to be monitored'][i] = (data if data != "" else 'nan')
    lines = str(row['Master Payload'])
    # data = data.strip()
    # data = data.strip('" \"')

    if(lines != "" or lines != 'nan'):
        #     handelNull()
        # else:
        with open('masterPayload.txt', 'w', encoding="utf-8") as f:
            f.write(lines)
        fileAnalyzer('masterPayload.txt', i)
    i = i+1

# for valName, valLen in masterDict.items():
#     print(valName, ": ", len(valLen))

for index, row in sheetDf1.iterrows():
    handelNull()
    data = str(row['Solution key'])
    data = data.strip()
    data = data.strip('" \"')

    masterDict['Solution Key'][i] = (data if data != "" else 'nan')
    # data = str(row['MESSAGE ID'])
    # data = data.strip()
    # data = data.strip('" \"')

    # data = tokennizeData(data)
    # data=sorted(data,key=str.casefold)
    # masterDict['PAF Error'][i] = (data if data != "" else 'nan')
    # data=str(row['Resolution Steps'])
    # masterDict['Resolution Steps'][i]=(data if data!="" else 'nan')
    # data=str(row['Resolution Strategy'])
    # masterDict['Resolution Strategy'][i]=(data if data!="" else 'nan')
    # data=str(row['Need to be monitored'])
    # masterDict['Need to be monitored'][i]=(data if data!="" else 'nan')
    lines = str(row['Incident Payload'])
    if(lines != "" or lines != 'nan'):
        #     handelNull()
        # else:
        with open('masterPayload.txt', 'w', encoding="utf-8") as f:
            f.write(lines)
        fileAnalyzer('masterPayload.txt', i)
    i = i+1

for valName, valLen in masterDict.items():
    print(valName, ": ", len(valLen))

ldf = pd.DataFrame(masterDict)
# Create a Pandas Excel writer using XlsxWriter as the engine.
writer = pd.ExcelWriter('Token1_Falcon_HC_Model_Data.xlsx', engine='xlsxwriter')

# Convert the dataframe to an XlsxWriter Excel object.
ldf.to_excel(writer, sheet_name='Model Data', index=False)

# Close the Pandas Excel writer and output the Excel file.
writer.close()