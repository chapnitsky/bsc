from django.shortcuts import render
from django.http import HttpResponse
import os

# Create your views here.

def home(request):
    print(f'{os.getcwd()}')
    return render(request, f'{os.getcwd()}\htmls\homelogin.html')
    # return HttpResponse('<h1>FUCK YOU</h1>')
