# not needed this doc
# Imports
import pandas as pd
from pymongo import MongoClient
import pandas as pd
import pymongo
import matplotlib.pyplot as plt 
import numpy as np
import base64
import io
# Load csv dataset
# data = pd.read_csv('train.csv')
# data = data.drop(data.index[30:42000])

# # making smaller the database
# # Connect to MongoDB
# client =  MongoClient("mongodb+srv://mon:key@cluster0.wm7jq.mongodb.net/mydb?retryWrites=true&w=majority")
# db = client['mydb']
# collection = db['test']
# data.reset_index(inplace=True)
# data_dict = data.to_dict("records")
# # Insert collection
# collection.insert_many(data_dict)

# plt.figure(figsize=(15,7))
# g = sns.countplot(Y_train, palette="icefire")
# plt.title("Number of digit classes")


def read_only(index):
    indexxx = int(index)

    client = MongoClient()
    #point the client at mongo URI
    client =  MongoClient("mongodb+srv://mon:key@cluster0.wm7jq.mongodb.net/mydb?retryWrites=true&w=majority")
    #select database
    findByID = { "index": indexxx }
    db = client['mydb']
    #select the collection within the database
    test = db.test
    #convert entire collection to Pandas dataframe
    test = pd.DataFrame(list(test.find(findByID)))
    return test



def read_data():
    # getting data from the database
    client = MongoClient()
    #point the client at mongo URI
    client =  MongoClient("mongodb+srv://mon:key@cluster0.wm7jq.mongodb.net/mydb?retryWrites=true&w=majority")
    #select database
    db = client['mydb']
    #select the collection within the database
    test = db.test
    #convert entire collection to Pandas dataframe
    test = pd.DataFrame(list(test.find()))
    df = test

    return df

    # DASH


def pixel(data):
    data.drop(data.columns[[0,1]], axis = 1, inplace = True)
    sample_size = data.shape[0] # Training set size
    validation_size = int(data.shape[0]*0.1) # Validation set si
    train_x = np.asarray(data.iloc[:sample_size-validation_size,1:]).reshape([sample_size-validation_size,28,28,1]) # taking all columns expect column 0
    buf = io.BytesIO()
    plt.imshow(train_x[0].reshape([28,28]),cmap="Blues") 
    plt.axis("off")
    plt.savefig("buf", format = "png")
    plt.close()
    data = base64.b64encode(buf.getbuffer()).decode("utf8")
    return data
