import pymongo

client = pymongo.MongoClient(
    "mongodb+srv://satlagamer:Swuiti25!!@website.7lapl.mongodb.net/myFirstDatabase?retryWrites=true&w=majority")
db = client.database
coll = db.data
