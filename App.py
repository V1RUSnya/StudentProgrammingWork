# This Python file uses the following encoding: utf-8
import sys
import os
import typing
import cv2
import imghdr
import pathlib
import matplotlib.pyplot as plt
import numpy as np
import idx2numpy
import time
from scipy import io as spio
from keras import optimizers
from keras import backend as K
from keras.datasets import cifar10
from keras.preprocessing import image
from keras.applications import ResNet50
from keras.applications.resnet import preprocess_input
from keras.models import Sequential, load_model, Model
from keras.layers import Dense, Dropout, Flatten, BatchNormalization, Activation, Reshape, LSTM, Input
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.optimizers import SGD, RMSprop, Adam
from keras.constraints import maxnorm
from keras.utils import np_utils, load_img, img_to_array ##
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
        self.setFixedSize(QSize(1600, 900))
        
        #Добавляем виджеты
        self.label = QLabel()
        self.button = QPushButton("Scan image")
        self.button.clicked.connect(self.get_photo)
        self.Mnistbutton = QPushButton("MNIST")
        self.Mnistbutton.clicked.connect(self.mnist_dataset)
        self.Kerasbutton = QPushButton("Generate cifar10 model")
        self.Kerasbutton.clicked.connect(self.KerasImage)
        self.ImagetoTextbutton = QPushButton("Scan text on image")
        self.ImagetoTextbutton.clicked.connect(self.imgtotext)
        self.TextGenbutton = QPushButton("Generate EMNIST model")
        self.TextGenbutton.clicked.connect(self.emnist)
        self.Matchbutton = QPushButton("Match Images")
        self.Matchbutton.clicked.connect(self.howmatch)
        
        #Создаем слой
        layout = QVBoxLayout()
        layout.addWidget(self.label)
        layout.addWidget(self.button)
        layout.addWidget(self.Kerasbutton)
        layout.addWidget(self.Mnistbutton)
        layout.addWidget(self.ImagetoTextbutton)
        layout.addWidget(self.TextGenbutton)
        layout.addWidget(self.Matchbutton)

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
        #file,src = QFileDialog.getOpenFileName(self, "Pick file", ".", "*.zip") # Запасной вариант
        #print(str(file))
        # Загрузка данных
        fashion_mnist = keras.datasets.fashion_mnist # Подгрузка с инета
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
        epochs = 100
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
        self.ImagetoTextbutton.setStyleSheet('background-color: #111111; color: white;')
        self.TextGenbutton.setStyleSheet('background-color: #111111; color: white;') 
        self.Matchbutton.setStyleSheet('background-color: #111111; color: white;')
    
    def emnist(self):
        folder_path = QFileDialog.getExistingDirectory(self, "Choose folder")
        print(folder_path)
        emnist_path = folder_path
        emnist_labels = [48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122]
        model = Sequential()
        model.add(Conv2D(filters=32, kernel_size=(3, 3), padding='valid', input_shape=(28, 28, 1), activation='relu'))
        model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        model.add(Flatten())
        model.add(Dense(512, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(len(emnist_labels), activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])
        X_train = idx2numpy.convert_from_file(emnist_path + '/emnist-byclass-train-images-idx3-ubyte')
        y_train = idx2numpy.convert_from_file(emnist_path + '/emnist-byclass-train-labels-idx1-ubyte')

        X_test = idx2numpy.convert_from_file(emnist_path + '/emnist-byclass-test-images-idx3-ubyte')
        y_test = idx2numpy.convert_from_file(emnist_path + '/emnist-byclass-test-labels-idx1-ubyte')

        X_train = np.reshape(X_train, (X_train.shape[0], 28, 28, 1))
        X_test = np.reshape(X_test, (X_test.shape[0], 28, 28, 1))

        print(X_train.shape, y_train.shape, X_test.shape, y_test.shape, len(emnist_labels))

        k = 1 # 1=100% 10=10% 100=1% Колво файлов в эпохе
        X_train = X_train[:X_train.shape[0] // k]
        y_train = y_train[:y_train.shape[0] // k]
        X_test = X_test[:X_test.shape[0] // k]
        y_test = y_test[:y_test.shape[0] // k]

        # Нормализуем
        X_train = X_train.astype(np.float32)
        X_train /= 255.0
        X_test = X_test.astype(np.float32)
        X_test /= 255.0

        x_train_cat = keras.utils.to_categorical(y_train, len(emnist_labels))
        y_test_cat = keras.utils.to_categorical(y_test, len(emnist_labels))
        # Снижаем скорость обучения
        learning_rate_reduction = keras.callbacks.ReduceLROnPlateau(monitor='val_accuracy', patience=3, verbose=1, factor=0.5, min_lr=0.00001)

        model.fit(X_train, x_train_cat, validation_data=(X_test, y_test_cat), callbacks=[learning_rate_reduction], batch_size=64, epochs=30)

        model.save('besedin_emnist.h5')

    def imgtotext(self):
        file,src = QFileDialog.getOpenFileName(self, "Pick file", ".", "*.jpg *.png *.jpeg") #Вызов проводника с аргументами 1-Текст, 2-Директория, 3-Формат
        ChooseFile = file
        #ChooseFile = '123.png'
        self.label.setPixmap(QPixmap(ChooseFile)) #Передаем путь в функцию Qpixmap и выводим результат через .setPixmap
        self.label.setScaledContents(True)
        print(ChooseFile)
        image_file = ChooseFile
        img = cv2.imread(image_file)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)
        img_erode = cv2.erode(thresh, np.ones((3, 3), np.uint8), iterations=1)
        emnist_labels = [48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122]

        # Получение контура
        contours, hierarchy = cv2.findContours(img_erode, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

        output = img.copy()

        for idx, contour in enumerate(contours):
            (x, y, w, h) = cv2.boundingRect(contour)
            if hierarchy[0][idx][3] == 0:
                cv2.rectangle(output, (x, y), (x + w, y + h), (70, 0, 0), 1)


        #cv2.imshow("Input", img)
        #cv2.imshow("Enlarged", img_erode)
        cv2.imshow("Output", output)
        cv2.waitKey(0)
        def letters_extract(image_file: str, out_size=28) -> list[any]:
            img = cv2.imread(image_file)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)
            img_erode = cv2.erode(thresh, np.ones((3, 3), np.uint8), iterations=1)

            # Получаем контуры
            contours, hierarchy = cv2.findContours(img_erode, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

            output = img.copy()

            letters = []
            for idx, contour in enumerate(contours):
                (x, y, w, h) = cv2.boundingRect(contour)
                if hierarchy[0][idx][3] == 0:
                    cv2.rectangle(output, (x, y), (x + w, y + h), (70, 0, 0), 1)
                    letter_crop = gray[y:y + h, x:x + w]
                    # print(letter_crop.shape)

                    # Ресайз букв до квадрата
                    size_max = max(w, h)
                    letter_square = 255 * np.ones(shape=[size_max, size_max], dtype=np.uint8)
                    if w > h:
                        # Увеличиваем изображение сверху вниз
                        y_pos = size_max//2 - h//2
                        letter_square[y_pos:y_pos + h, 0:w] = letter_crop
                    elif w < h:
                        # Увеличиваем слева направо
                        x_pos = size_max//2 - w//2
                        letter_square[0:h, x_pos:x_pos + w] = letter_crop
                    else:
                        letter_square = letter_crop

                    # Изменяем размер буквы до 28x28 и добавляем букву и ее координату X
                    letters.append((x, w, cv2.resize(letter_square, (out_size, out_size), interpolation=cv2.INTER_AREA)))

            # Сортируем по координате Х
            letters.sort(key=lambda x: x[0], reverse=False)

            return letters
        letters = letters_extract(image_file)
        cv2.imshow("0", letters[0][2])
        cv2.imshow("1", letters[1][2])
        cv2.imshow("2", letters[2][2])
        cv2.imshow("3", letters[3][2])
        cv2.imshow("4", letters[4][2])
        cv2.waitKey(0)
        
        model = keras.models.load_model('besedin_emnist.h5') # Загружаем модель
        
        def emnist_predict_img(model, img):
            img_arr = np.expand_dims(img, axis=0)
            img_arr = 1 - img_arr/255.0
            img_arr[0] = np.rot90(img_arr[0], 3)
            img_arr[0] = np.fliplr(img_arr[0])
            img_arr = img_arr.reshape((1, 28, 28, 1))

            predict = model.predict([img_arr])
            result = np.argmax(predict, axis=1)
            return chr(emnist_labels[result[0]])
        def img_to_str(model: any, image_file: str):
            letters = letters_extract(image_file)
            s_out = ""
            for i in range(len(letters)):
                dn = letters[i+1][0] - letters[i][0] - letters[i][1] if i < len(letters) - 1 else 0
                s_out += emnist_predict_img(model, letters[i][2])
                if (dn > letters[i][1]/4):
                    s_out += ' '
            return s_out
        s_out = img_to_str(model, ChooseFile)
        print(s_out)
        
    def howmatch(self):
        base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3)) # Создаем базовую модель ResNet50
        # Добавляем слои для определения схожести изображений
        input_a = Input(shape=(224, 224, 3))
        input_b = Input(shape=(224, 224, 3))
        processed_a = base_model(input_a)
        processed_b = base_model(input_b)

        flatten_a = Flatten()(processed_a)
        flatten_b = Flatten()(processed_b)

        merged_vector = tf.keras.layers.concatenate([flatten_a, flatten_b])

        dense1 = Dense(256, activation='relu')(merged_vector)
        output = Dense(1, activation='sigmoid')(dense1)

        model = Model(inputs=[input_a, input_b], outputs=output)
        
        # Компилируем модель
        model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=0.0001), metrics=['accuracy'])
        
        # Загружаем и предварительно обрабатываем изображения
        def load_and_preprocess_image(image_path):
            img = load_img(image_path, target_size=(224, 224))
            img = img_to_array(img)
            img = preprocess_input(img)
            return img
        # Примеры путей к изображениям
        image_path1,musor = QFileDialog.getOpenFileName(self, "Pick file", ".", "*.jpg *.png *.jpeg")
        image_path2,musor = QFileDialog.getOpenFileName(self, "Pick file", ".", "*.jpg *.png *.jpeg")
        self.label.setPixmap(QPixmap(image_path1)) #Передаем путь в функцию Qpixmap и выводим результат через .setPixmap
        self.label.setScaledContents(True)
        # Загружаем и предварительно обрабатываем изображения
        img1 = load_and_preprocess_image(image_path1)
        img2 = load_and_preprocess_image(image_path2)
        # Размер пакета (batch size) для предсказания
        batch_size = 1
        # Делаем предсказание
        result = model.predict([np.array([img1]), np.array([img2])], batch_size=batch_size)
        # Если result[0][0] близко к 1, то изображения считаются схожими, если близко к 0 - несхожими.
        print(f'Photos are similar to {result[0][0]}')
        

def application():
    app = QApplication(sys.argv) #Вызываем окно с доступом к консоли (sys.argv)
    window = Start() #Выполняем класс
    window.show() #Изначально окно скрыто, потому показываем
    sys.exit(app.exec()) #Запускаем цикл событий

print("TensorFlow ", tf.__version__)
print("GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
application()
