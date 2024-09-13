from flask import Flask, request, render_template, redirect, url_for
import joblib
import numpy as np

app = Flask(__name__)

# Downloading the model
model = joblib.load('models/iris_classifier.pkl')

# Main page
@app.route('/')
def index():
    return render_template('index.html')

# Processing of form data
@app.route('/predict', methods=['POST', "GET"])

def predict():
    if request.method == "GET":
        return render_template('index.html')
    
    if request.method == "POST":
        try:
            # Retrieving data from a form
            sepal_length = float(request.form['sepal_length'])
            sepal_width = float(request.form['sepal_width'])
            petal_length = float(request.form['petal_length'])
            petal_width = float(request.form['petal_width'])

            # Checking the range of values
            if sepal_length <= 0 or sepal_width <= 0 or petal_length <= 0 or petal_width <= 0:
                raise ValueError("All measurements must be positive numbers.")

            # Convert data to the format required by the model
            features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])

            # Prediction
            prediction = model.predict(features)
            iris_species = ['setosa', 'versicolor', 'virginica']
            result = iris_species[prediction[0]]

            return render_template('result.html', result=result)

        except ValueError as e:
            # Handling input error and redirecting back to the main page with an error message
            error_message = str(e)
            return render_template('index.html', error=error_message)
        except Exception as e:
            # Handling other errors
            error_message = "An unexpected error occurred: " + str(e)
            return render_template('index.html', error=error_message)

if __name__ == '__main__':
    app.run(debug=True)

