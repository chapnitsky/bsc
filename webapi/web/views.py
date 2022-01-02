from django.shortcuts import render
from django.http import HttpResponse
import os


# Create your views here.

def home(request):
    print(f'{os.getcwd()}\web\htmls\homelogin.html\n')
    return render(request, f'{os.getcwd()}\web\htmls\homelogin.html')
    # return HttpResponse('<h1>FUCK YOU</h1>')


def sub(request):
    return render(request, f'{os.getcwd()}\web\htmls\home.html')
