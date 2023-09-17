from flask import Flask, render_template, request,jsonify
import numpy as np
import pandas as pd
import joblib
from app import FeatureHashPreprocessor, CategoryPreprocessor, StandardScaler_
from sklearn.pipeline import Pipeline
from flask_cors import CORS 
import json

app = Flask(__name__)

CORS(app)

pipeline = joblib.load("pipeline.pkl")

@app.route('/')
def home():
    
    return render_template('index.html')
@app.route('/predict',methods=['POST'])
def predict():
    if request.method == 'POST':
        data = request.get_json()
        X = pd.DataFrame({
            "EaseofUse": data["eou"],
            "Satisfaction": data["satisfaction"],
            "Age": data["age"],
            "Condition": data["condtion"],
            "Sex": data["sex"]
        }, index=[0])       
        y_pred = pipeline.predict(X)
        print('Noel',y_pred)

        
        return json.dumps({'y_pred': int(y_pred[0])})
    
   
    
   
if __name__=='__main__':
    app.run(port=8000)