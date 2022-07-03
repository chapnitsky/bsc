from django.shortcuts import render, redirect
from django.http import HttpResponse
from bson.objectid import ObjectId
from collections import namedtuple
import pymongo

client = pymongo.MongoClient(
    f'mongodb+srv://satlagamer:Swuiti25!!@website.7lapl.mongodb.net/myFirstDatabase?retryWrites=true&w=majority')
db = client.database
coll = db.data


def login(request):
    return render(request, 'web/login.html', {"Title": "Login", "act": "/sub", "h1_base": "Login"})


def decide_action(request, string):
    decision = string[-1]

    filt = {"_id": ObjectId(string[:-1])}
    new_val = None
    if decision == '!':
        text = request.GET.get('fix_correct')
        if text:
            print(f'text : {text}')
            new_val = {"$set": {"ans": 1, "checked": True, "sentence": str(text)}}
        else:
            new_val = {"$set": {"ans": 1, "checked": True}}
    elif decision == '+':
        y = coll.find_one(filt)
        y['id'] = str(y.pop('_id'))
        object_name = namedtuple("Sentence", y.keys())(*y.values())
        return render(request, 'web/home.html',
                      {"Title": "Home", "act": "", "h1_base": "Home", "sentences": object_name, "correct": True})

    elif decision == '-':
        new_val = {"$set": {"ans": 0, "checked": True}}

    coll.update_one(filt, new_val)
    return redirect('/sub')


def checker(request):
    res = coll.find_one({"checked": False})
    if not res:
        return redirect('/sub')
    y = res
    y['id'] = str(y.pop('_id'))
    object_name = namedtuple("Sentence", y.keys())(*y.values())

    return render(request, 'web/home.html', {"Title": "Home", "act": "", "h1_base": "Home", "sentences": object_name})


def home(request):
    usrname = request.GET.get('uname')
    password = request.GET.get('psw')

    # print(f'usr: {usrname}\npsw: {password}')
    return render(request, 'web/home.html', {"Title": "Home", "act": "/checker", "h1_base": "Home"})
