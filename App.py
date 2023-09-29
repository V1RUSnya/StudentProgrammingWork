# This Python file uses the following encoding: utf-8
import sys
import os
import typing
import matplotlib.pyplot as plt
import numpy as np
from keras.datasets import cifar10
from keras.preprocessing import image
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Flatten, BatchNormalization, Activation
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.constraints import maxnorm
from keras.utils import np_utils
from keras.metrics import Accuracy
import tensorflow as tf
from tensorflow import keras
from PyQt6 import QtCore
from PyQt6.QtGui import QPixmap
from PyQt6.QtCore import QFileInfo, QSize
from PyQt6.QtWidgets import QFileDialog, QWidget, QVBoxLayout, QLabel, QPushButton, QMainWindow, QApplication, QMessageBox

class Start(QMainWindow):
    def __init__(self):
        super().__init__()
        
        self.setWindowTitle("BesDV") #Имя программы
        self.setFixedSize(QSize(800, 500))
        
        #Добавляем виджеты
        self.label = QLabel()
        self.button = QPushButton("Scan image")
        self.button.clicked.connect(self.get_photo)
        self.Mnistbutton = QPushButton("MNIST")
        self.Mnistbutton.clicked.connect(self.mnist_dataset)
        self.Kerasbutton = QPushButton("Generate model")
        self.Kerasbutton.clicked.connect(self.KerasImage)
        
        #Создаем слой
        layout = QVBoxLayout()
        layout.addWidget(self.label)
        layout.addWidget(self.button)
        layout.addWidget(self.Kerasbutton)
        layout.addWidget(self.Mnistbutton)

        #Создание макета
        self.container = QWidget()
        self.container.setLayout(layout)
        self.style() #Метод включающий стили

        
        self.setCentralWidget(self.container) #Устанавливаем центральный виджет
        
    def get_photo(self):
        file,src = QFileDialog.getOpenFileName(self, "Pick file", ".", "*.jpg *.png *.jpeg") #Вызов проводника с аргументами 1-Текст, 2-Директория, 3-Формат
        ChooseFile = str(QFileInfo(file).path()) + '/' + str(QFileInfo(file).fileName()) #Соединяем путь и имя файла
        print(ChooseFile)
        if ChooseFile == '/':
            self.label.setText(ChooseFile + ' is not correct path!') #Передаем путь файла в виде текста
            Error = QMessageBox()
            Error.setWindowTitle('Error')
            Error.setText('Invalid path!')
            #Error.setIcon(QMessageBox.warning) Почему то не работает))
            Error.exec()
        else:
            self.label.setPixmap(QPixmap(ChooseFile)) #Передаем путь в функцию Qpixmap и выводим результат через .setPixmap
            self.label.setScaledContents(True)
            model = load_model('Besedin_model.h5') # Загрузка модели
            img_path = ChooseFile
            img = tf.keras.preprocessing.image.load_img(img_path, target_size=(32, 32))  # Задайте размер, соответствующий вашей модели
            img_array = tf.keras.preprocessing.image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array = img_array / 255.0  # Нормализация изображения, как и при обучении
            # Выполнение предсказания
            predictions = model.predict(img_array)
            # Вывод результатов
            class_labels = ["Airplane", "Automobile", "Bird", "Cat", "Deer", "Dog", "Frog", "Horse", "Ship", "Truck"]
            predicted_class = np.argmax(predictions)
            predicted_label = class_labels[predicted_class]

            print(f"Its may be: {predicted_label}")
        
    def mnist_dataset(self):
        file,src = QFileDialog.getOpenFileName(self, "Pick file", ".", "*.zip")
        print(str(file))
        # Загрузка данных
        fashion_mnist = keras.datasets.fashion_mnist
        (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

        # Нормализация данных (масштабирование значений в диапазоне [0, 1])
        train_images = train_images / 255.0
        test_images = test_images / 255.0
        model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),  # Преобразование 28x28 в 784 пикселя
    keras.layers.Dense(128, activation='relu'),  # Полносвязный слой с функцией активации ReLU
    keras.layers.Dense(10, activation='softmax')  # Выходной слой с 10 нейронами для 10 классов и функцией активации softmax
])
        model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',  # Функция потерь для задачи классификации
              metrics=['accuracy'])
        model.fit(train_images, train_labels, epochs=10, validation_split=0.2)
        test_loss,test_acc = model.evaluate(test_images, test_labels)
        print(f'Test accuracy: {test_acc}')
        result = QMessageBox()
        result.setWindowTitle('Accuracy')
        result.setText(f'Accuracy is {test_acc}')
        #result.exec()
        
        # Загрузка классов Fashion MNIST
        class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']


        # Сделать прогнозы для тестовых изображений
        predictions = model.predict(test_images)

        num_rows = 5
        num_cols = 5
        num_images = num_rows * num_cols

        plt.figure(figsize=(12, 10))

        for i in range(num_images):
            plt.subplot(num_rows, num_cols, i + 1)
            plt.imshow(test_images[i], cmap='binary')
            predicted_label = np.argmax(predictions[i])
            true_label = test_labels[i]
            if predicted_label == true_label:
                color = 'green'
            else:
                color = 'red'
            plt.xlabel(f'{class_names[predicted_label]} ({class_names[true_label]})', color=color)
            plt.xticks([])
            plt.yticks([])

        plt.tight_layout()
        plt.show()
        
    def KerasImage(self):
        # Рандомим сид
        seed = 21
        (X_train, y_train), (X_test, y_test) = cifar10.load_data() # Загружаем данные
        # Нормализуем данные (Тупо делим на 255 xD )
        X_train = X_train.astype('float32')
        X_test = X_test.astype('float32')
        X_train = X_train / 255.0
        X_test = X_test / 255.0
        # Указываем кол-во классов в наборе данных (До скольки нейронов сжимать конечный слой)
        y_train = np_utils.to_categorical(y_train)
        y_test = np_utils.to_categorical(y_test)
        class_num = y_test.shape[1]
        
        model = Sequential() # Создание модели
        ## Сверточные слои
        model.add(Conv2D(32, (3, 3), input_shape=X_train.shape[1:], padding='same'))
        model.add(Activation('relu'))
        model.add(Conv2D(32, (3, 3), input_shape=(3, 32, 32), activation='relu', padding='same'))
        model.add(Dropout(0.2))
        model.add(BatchNormalization())
        model.add(Conv2D(64, (3, 3), padding='same'))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.2))
        model.add(BatchNormalization())
        ## Повторяем слои для лучшего результата
        model.add(Conv2D(64, (3, 3), padding='same'))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.2))
        model.add(BatchNormalization())
        model.add(Conv2D(128, (3, 3), padding='same'))
        model.add(Activation('relu'))
        model.add(Dropout(0.2))
        model.add(BatchNormalization())
        # Слой исключения
        model.add(Flatten())
        model.add(Dropout(0.2))
        #Создаем первый плотно связанный слой
        model.add(Dense(256, kernel_constraint=maxnorm(3)))
        model.add(Activation('relu'))
        model.add(Dropout(0.2))
        model.add(BatchNormalization())
        model.add(Dense(128, kernel_constraint=maxnorm(3)))
        model.add(Activation('relu'))
        model.add(Dropout(0.2))
        model.add(BatchNormalization())
        #Функция активации softmax выбирает нейрон с наибольшей вероятностью в качестве выходного значения
        model.add(Dense(class_num))
        model.add(Activation('softmax'))
        # Наконец то компилируем
        # Алгоритм Адама является одним из наиболее часто используемых оптимизаторов, потому что он дает высокую производительность в большинстве задач
        epochs = 25
        optimizer = 'adam'
        model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
        #print(model.summary()) # Вывод данных для отладки
        # Обучение модели
        np.random.seed(seed)
        model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=epochs, batch_size=64)
        ## Оценка модели
        scores = model.evaluate(X_test, y_test, verbose=0)
        print("Accuracy: %.2f%%" % (scores[1]*100))
        model.save('Besedin_model.h5')
        
    def style(self):
        self.container.setStyleSheet('background-color: #373737;')
        self.button.setStyleSheet('background-color: #111111; color: white;')
        self.Kerasbutton.setStyleSheet('background-color: #111111; color: white;')
        self.Mnistbutton.setStyleSheet('background-color: #111111; color: white;')

def application():
    app = QApplication(sys.argv) #Вызываем окно с доступом к консоли (sys.argv)
    window = Start() #Выполняем класс
    window.show() #Изначально окно скрыто, потому показываем
    sys.exit(app.exec()) #Запускаем цикл событий

print("TensorFlow ", tf.__version__)
print("GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
application()
