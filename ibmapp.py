import numpy as np
import pickle
import joblib
import matplotlib
import matplotlib.pyplot as plt
import time
import pandas
import os
from flask import Flask, request, jsonify, render_template
import json

import requests

# NOTE: you must manually set API_KEY below using information retrieved from your IBM Cloud account.
API_KEY = "l0ArGgOyZ3AWyBtW5SQG-9PPK8SgWoamljoLcqyw3CaR"
token_response = requests.post('https://iam.cloud.ibm.com/identity/token', data={"apikey": API_KEY, "grant_type": 'urn:ibm:params:oauth:grant-type:apikey'})
mltoken = token_response.json()["access_token"]

header = {'Content-Type': 'application/json', 'Authorization': 'Bearer ' + mltoken}

# NOTE: manually define and pass the array(s) of values to be scored in the next line
#payload_scoring = {"input_data": [{"fields": [array_of_input_fields], "values": [array_of_values_to_be_scored, another_array_of_values_to_be_scored]}]}

#response_scoring = requests.post('https://us-south.ml.cloud.ibm.com/ml/v4/deployments/8e16b491-9bb5-40c2-9ded-a3094b4de776/predictions?version=2021-11-10', json=payload_scoring, headers={'Authorization': 'Bearer ' + mltoken})
#print("Scoring response")
#print(response_scoring.json())


app = Flask(__name__)

scale = pickle.load(open(r"C:\Users\manish\Desktop\project\Dynamic-Price-Predictor-for-cabs-main\Dynamic Price Prediction for cabs main\Flask\model.pkl",'rb'))

@app.route('/')# route to display the home page
def home():
    return render_template('index.html') #rendering the home page
@app.route('/Prediction',methods=['POST','GET'])
def prediction(): # route which will take you to the prediction page
    return render_template('index1.html')
@app.route('/Home',methods=['POST','GET'])
def my_home():
    return render_template('index.html')

@app.route('/predict',methods=["POST","GET"])# route to show the predictions in a web UI
def predict():
    #return render_template('index1.html')
    #  reading the inputs given by the user
    cab_type = request.form["cab_type"]
    name = request.form["name"]
    product_id = request.form["product_id"]
    source = request.form["source"]
    destination = request.form["destination"]
    

    t = [[int(cab_type),int(name),int(product_id),int(source),int(destination)]]
    payload_scoring = {"input_data": [{"field": [['cab_type', 'name', 'product_id', 'source', 'destination']], "values": t}]}

    response_scoring = requests.post('https://us-south.ml.cloud.ibm.com/ml/v4/deployments/9e32df44-c7b2-4725-92c3-059c280aabdc/predictions?version=2021-11-04', json=payload_scoring, headers={'Authorization': 'Bearer ' + mltoken})
    print("Scoring response")
    predictions = response_scoring.json()
    pred = predictions['predictions'][0]['values'][0][0]
    print(pred)
   # if(pred=="Yes"):
        
    return render_template('result.html',prediction=pred)
    #else:
     # predictions using the loaded model file
       # return render_template('index.html')
     # showing the prediction results in a UI
if __name__=="__main__":
    
    # app.run(host='0.0.0.0', port=8000,debug=True)    # running the app
    port=int(os.environ.get('PORT',5000))
    app.run(port=port,debug=True,use_reloader=False)