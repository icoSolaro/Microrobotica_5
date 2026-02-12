from flask import Flask, render_template, request
from joblib import load
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler

model_labelencoder = load("./modelli/model_labelencoder.dump")
regressor = load("./modelli/regressor.dump")
trasmission_labelencoder = load("./modelli/trasmission_labelencoder.dump")


app = Flask(__name__)

@app.route("/", methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        model = request.form['model']
        engine_power = int(request.form['engine_power'])
        transmission = request.form['transmission']
        age_in_days = int(request.form['age_in_days'])
        km = int(request.form['km'])
        previous_owners = int(request.form['previous_owners'])
        lat = float(request.form['lat'])
        lon = float(request.form['lon'])
        model_numeric=model_labelencoder.transform([model])[0]
        trasmission_numeric=trasmission_labelencoder.transform([transmission])[0]
        fiat_500=np.array([[engine_power, age_in_days, km,previous_owners, lat, lon, model_numeric,trasmission_numeric]])
        price=regressor.predict(fiat_500)
        
        print(price)

        return render_template('prezzo.html', price=np.round(price[0],2))

    elif request.method == 'GET':
        return render_template('index.html')
    

if __name__ == '__main__':
    app.run(host='localhost')
