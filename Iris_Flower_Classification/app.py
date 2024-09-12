from flask import Flask, request, render_template
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
@app.route('/predict', methods=['POST'])
def predict():
    # Получение данных из формы
    sepal_length = float(request.form['sepal_length'])
    sepal_width = float(request.form['sepal_width'])
    petal_length = float(request.form['petal_length'])
    petal_width = float(request.form['petal_width'])

    # Преобразование данных в формат, требуемый моделью
    features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    
    # Предсказание
    prediction = model.predict(features)
    iris_species = ['setosa', 'versicolor', 'virginica']
    result = iris_species[prediction[0]]
    
    return render_template('result.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)
