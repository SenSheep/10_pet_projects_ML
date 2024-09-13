from flask import Flask, request, render_template, redirect, url_for
import joblib
import numpy as np

app = Flask(__name__)

# Загрузка модели
model = joblib.load('models/iris_classifier.pkl')

# Главная страница
@app.route('/')
def index():
    return render_template('index.html')

# Обработка данных формы
@app.route('/predict', methods=['POST', "GET"])

def predict():
    if request.method == "GET":
        return render_template('index.html')
    
    if request.method == "POST":
        try:
            # Получение данных из формы
            sepal_length = float(request.form['sepal_length'])
            sepal_width = float(request.form['sepal_width'])
            petal_length = float(request.form['petal_length'])
            petal_width = float(request.form['petal_width'])

            # Проверка диапазона значений
            if sepal_length <= 0 or sepal_width <= 0 or petal_length <= 0 or petal_width <= 0:
                raise ValueError("All measurements must be positive numbers.")

            # Преобразование данных в формат, требуемый моделью
            features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])

            # Предсказание
            prediction = model.predict(features)
            iris_species = ['setosa', 'versicolor', 'virginica']
            result = iris_species[prediction[0]]

            return render_template('result.html', result=result)

        except ValueError as e:
            # Обработка ошибки ввода и перенаправление обратно на главную страницу с сообщением об ошибке
            error_message = str(e)
            return render_template('index.html', error=error_message)
        except Exception as e:
            # Обработка других ошибок
            error_message = "An unexpected error occurred: " + str(e)
            return render_template('index.html', error=error_message)

if __name__ == '__main__':
    app.run(debug=True)

