import pymongo
from sensitive import username, pswd


client = pymongo.MongoClient(
    f'mongodb+srv://{username}:{pswd}@website.7lapl.mongodb.net/myFirstDatabase?retryWrites=true&w=majority')
db = client.database
coll = db.data
