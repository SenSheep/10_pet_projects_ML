import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt

# Загрузка набора данных MNIST
mnist = tf.keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# Нормализация данных (приведение значений пикселей к диапазону от 0 до 1)
train_images, test_images = train_images / 255.0, test_images / 255.

model = models.Sequential([
    layers.Flatten(input_shape=(28, 28)),  # Преобразование 28x28 изображения в одномерный вектор
    layers.Dense(128, activation='relu'),  # Полносвязанный слой с 128 нейронами
    layers.Dense(10, activation='softmax')  # Выходной слой для 10 классов (цифры 0-9)
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(train_images, train_labels, epochs=5)

test_loss, test_acc = model.evaluate(test_images, test_labels)
print(f'Точность на тестовых данных: {test_acc}')

predictions = model.predict(test_images)
predictions.to_csv('/', index=False)

# Визуализация первых 5 изображений и их предсказаний
for i in range(len(test_images)):
    plt.imshow(test_images[i], cmap=plt.cm.binary)
    plt.title(f"Предсказано: {predictions[i].argmax()}, Истинное: {test_labels[i]}")
    plt.show()
