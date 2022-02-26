from django.shortcuts import render
from django.http import HttpResponse
import os
import pymongo as mongo


cli = mongo.MongoClient("mongodb://localhost:27017/")
db = cli.database
coll = db.data


def home(request):
    # coll.insert_one({"_id": 0, "name": "fss"})
    # print(f'{os.getcwd()}\web\htmls\homelogin.html\n')
    print("homed\n")

    return render(request, f'{os.getcwd()}\web\htmls\homelogin.html')


def checker(request):
    res = coll.find({})
    for x in res:
        print(f'document: {x}\n')

    return render(request, f'{os.getcwd()}\web\htmls\home.html')


def sub(request):
    res = coll.count_documents({})
    print(f'counted: {res}')

    return render(request, f'{os.getcwd()}\web\htmls\home.html')
