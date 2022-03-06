from django.shortcuts import render, redirect
from django.http import HttpResponse
from mongo import coll
from bson.objectid import ObjectId
from collections import namedtuple


def login(request):
    return render(request, 'web/login.html', {"Title": "Login", "act": "/sub", "h1_base": "Login"})


def checked(request, string):
    decision = string[-1]
    print(f'string: {string[:-1]}')

    filt = {"_id": ObjectId(string[:-1])}
    new_val = {"$set": {"ans": 1 if decision == '+' else -1, "checked": True}}
    coll.update_one(filt, new_val)
    return redirect('/sub')


def checker(request):
    res = coll.find_one({"checked": False})
    if not res:
        return redirect('/sub')
    y = res
    y['id'] = str(y.pop('_id'))
    object_name = namedtuple("Sentence", y.keys())(*y.values())
    print(f'document: {object_name}\n')

    return render(request, 'web/home.html', {"Title": "Home", "act": "", "h1_base": "Home", "sentences": object_name})


def home(request):
    return render(request, 'web/home.html', {"Title": "Home", "act": "/checker", "h1_base": "Home"})
