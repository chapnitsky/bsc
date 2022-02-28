from django.shortcuts import render
from django.http import HttpResponse
import os
import pymongo as mongo
from collections import namedtuple

cli = mongo.MongoClient("mongodb://localhost:27017/")
db = cli.database
coll = db.data


def login(request):
    # coll.insert_one({"_id": 0, "name": "fss"})
    # print(f'{os.getcwd()}\web\htmls\homelogin.html\n')
    print("homed")

    return render(request, 'web/login.html', {"Title": "Login", "act": "/sub", "h1_base": "Login"})


def checker(request):
    res = coll.find({})
    unchecked = []
    for x in res:
        y = x
        y['id'] = y.pop('_id')
        if y['checked']:
            continue
        # y.pop('_id', None)  # For mongo o
        object_name = namedtuple("aaa", y.keys())(*y.values())

        print(type(object_name))
        print(f'document: {object_name}\n')
        unchecked.append(object_name)

    return render(request, 'web/home.html', {"Title": "Home", "act": "", "h1_base": "Home", "sentences": unchecked})


def home(request):
    res = coll.count_documents({})
    print(f'counted: {res}')

    return render(request, 'web/home.html', {"Title": "Home", "act": "/checker", "h1_base": "Home"})
