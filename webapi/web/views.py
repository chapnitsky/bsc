from django.shortcuts import render, redirect
from django.http import HttpResponse
import os
import pymongo as mongo
from collections import namedtuple

cli = mongo.MongoClient("mongodb://localhost:27017/")
db = cli.database
coll = db.data


def login(request):
    return render(request, 'web/login.html', {"Title": "Login", "act": "/sub", "h1_base": "Login"})


def checked(request, string):
    print(f'string: {string}')
    return redirect('/sub')


def checker(request):
    res = coll.find_one({"checked": False})
    y = res
    y['id'] = str(y.pop('_id'))
    object_name = namedtuple("Sentence", y.keys())(*y.values())
    print(f'document: {object_name}\n')

    return render(request, 'web/home.html', {"Title": "Home", "act": "", "h1_base": "Home", "sentences": object_name})


def home(request):
    return render(request, 'web/home.html', {"Title": "Home", "act": "/checker", "h1_base": "Home"})
