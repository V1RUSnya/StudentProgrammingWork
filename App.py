# This Python file uses the following encoding: utf-8
import sys
import typing
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
        self.button = QPushButton("Image")
        self.button.clicked.connect(self.get_photo)
        self.Mnistbutton = QPushButton("MNIST")
        self.Mnistbutton.clicked.connect(self.mnist_dataset)
        
        #Создаем слой
        layout = QVBoxLayout()
        layout.addWidget(self.label)
        layout.addWidget(self.button)
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
        result.exec()

        
    def style(self):
        self.container.setStyleSheet('background-color: #373737;')
        self.button.setStyleSheet('background-color: #111111; color: white;')
        self.Mnistbutton.setStyleSheet('background-color: #111111; color: white;')

def application():
    app = QApplication(sys.argv) #Вызываем окно с доступом к консоли (sys.argv)
    window = Start() #Выполняем класс
    window.show() #Изначально окно скрыто, потому показываем
    sys.exit(app.exec()) #Запускаем цикл событий

print("TensorFlow ", tf.__version__)
print("GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
application()
