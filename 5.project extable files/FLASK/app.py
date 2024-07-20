from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)
model = joblib.load('flights.pkl')

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        try:
            flightnumber = int(request.form['flightnumber'])
            month = int(request.form['MONTH'])
            day_of_month = int(request.form['DAY_OF_MONTH'])
            day_of_week = int(request.form['DAY_OF_WEEK'])

            origin = request.form['carrier']
            origin_map = {"MSP": 1, "DTW": 2, "JFK": 3, "SEA": 4, "ATL": 5}
            origin = origin_map.get(origin, 0)

            destination = request.form['dest']
            destination_map = {"MSP": 1, "DTW": 2, "JFK": 3, "SEA": 4, "ATL": 5}
            destination = destination_map.get(destination, 0)

            dept = int(request.form['DEP_DEL15'])
            arr_time = int(request.form['CRS_ARR_TIME'])
            act_dept = int(request.form['ARR_DEL15'])
            dept15 = dept - act_dept

            total = np.array([[month, day_of_month, day_of_week, origin, destination, dept, arr_time, dept15]])

            # Debug prints
            print("Input values:", total)

            y_pred = model.predict(total)

            # Debug print for prediction result
            print("Prediction result:", y_pred)

            ans = 'The Flight will be on time' if y_pred[0] == 0 else 'The Flight will be delayed'

            return render_template('predict.html', showcase=ans)
        except Exception as e:
            print("Error:", e)  # Debug print for error
            return str(e)
    return render_template('predict.html', showcase=None)

if __name__ == '__main__':
    app.run(debug=True)
