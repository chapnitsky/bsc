from django.shortcuts import render
from django.http import HttpResponse
import os
import pymongo as mongo

# Create your views here.

cli = mongo.MongoClient("mongodb://localhost:27017/")
db = cli.database
coll = db.data


def home(request):
    # coll.insert_one({"_id": 0, "name": "fuck"})
    # print(f'{os.getcwd()}\web\htmls\homelogin.html\n')
    return render(request, f'{os.getcwd()}\web\htmls\homelogin.html')
    # return HttpResponse('<h1>FUCK YOU</h1>')


def checker(request):
    print("worked\n")
    return render(request, f'{os.getcwd()}\web\htmls\home.html')


def sub(request):
    res = coll.count_documents({})
    print(f'counted: {res}')
    return render(request, f'{os.getcwd()}\web\htmls\home.html')
