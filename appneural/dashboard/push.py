# not needed this doc
# Imports
import pandas as pd
from pymongo import MongoClient
# Load csv dataset
data = pd.read_csv('train.csv')
# Connect to MongoDB
client =  MongoClient("mongodb+srv://mon:key@cluster0.wm7jq.mongodb.net/mydb?retryWrites=true&w=majority")
db = client['mydb']
collection = db['numbers']
data.reset_index(inplace=True)
data_dict = data.to_dict("records")
# Insert collection
collection.insert_many(data_dict)