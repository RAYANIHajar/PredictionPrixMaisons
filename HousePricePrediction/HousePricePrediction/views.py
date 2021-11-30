from django.shortcuts import render;
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
import csv
def home(request):
    return render(request, "home.html")

def predict(request):
    return render(request, "predict.html")

def result(request):
    data = pd.read_csv("C:/Users/HP/Downloads/Compressed/USA_Housing.csv")
    data = data.drop(["Address"], axis=1)
    x = data.drop("Price", axis=1)
    y = data["Price"]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.30)
    model = LinearRegression()
    model.fit(x_train, y_train)

    var1 = float(request.GET['n1'])
    var2 = float(request.GET['n2'])
    var3 = float(request.GET['n3'])
    var4 = float(request.GET['n4'])
    var5 = float(request.GET['n5'])

    pred = model.predict(np.array([var1, var2, var3, var4, var5]).reshape(1, -1))
    pred = round(pred[0])
    price = "Le prix prévu est : $" + str(pred)

    return render(request, 'predict.html', {"result2": price})