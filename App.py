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
        
        self.setWindowTitle("BesDV") #��� ���������
        self.setFixedSize(QSize(800, 500))
        
        #��������� �������
        self.label = QLabel()
        self.button = QPushButton("Image")
        self.button.clicked.connect(self.get_photo)
        self.Mnistbutton = QPushButton("MNIST")
        self.Mnistbutton.clicked.connect(self.mnist_dataset)
        
        #������� ����
        layout = QVBoxLayout()
        layout.addWidget(self.label)
        layout.addWidget(self.button)
        layout.addWidget(self.Mnistbutton)

        #�������� ������
        self.container = QWidget()
        self.container.setLayout(layout)
        self.style() #����� ���������� �����

        
        self.setCentralWidget(self.container) #������������� ����������� ������
        
    def get_photo(self):
        file,src = QFileDialog.getOpenFileName(self, "Pick file", ".", "*.jpg *.png *.jpeg") #����� ���������� � ����������� 1-�����, 2-����������, 3-������
        ChooseFile = str(QFileInfo(file).path()) + '/' + str(QFileInfo(file).fileName()) #��������� ���� � ��� �����
        print(ChooseFile)
        if ChooseFile == '/':
            self.label.setText(ChooseFile + ' is not correct path!') #�������� ���� ����� � ���� ������
            Error = QMessageBox()
            Error.setWindowTitle('Error')
            Error.setText('Invalid path!')
            #Error.setIcon(QMessageBox.warning) ������ �� �� ��������))
            Error.exec()
        else:
            self.label.setPixmap(QPixmap(ChooseFile)) #�������� ���� � ������� Qpixmap � ������� ��������� ����� .setPixmap
            self.label.setScaledContents(True)
        
    def mnist_dataset(self):
        file,src = QFileDialog.getOpenFileName(self, "Pick file", ".", "*.zip")
        print(str(file))
        # �������� ������
        fashion_mnist = keras.datasets.fashion_mnist
        (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

        # ������������ ������ (��������������� �������� � ��������� [0, 1])
        train_images = train_images / 255.0
        test_images = test_images / 255.0
        model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),  # �������������� 28x28 � 784 �������
    keras.layers.Dense(128, activation='relu'),  # ������������ ���� � �������� ��������� ReLU
    keras.layers.Dense(10, activation='softmax')  # �������� ���� � 10 ��������� ��� 10 ������� � �������� ��������� softmax
])
        model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',  # ������� ������ ��� ������ �������������
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
    app = QApplication(sys.argv) #�������� ���� � �������� � ������� (sys.argv)
    window = Start() #��������� �����
    window.show() #���������� ���� ������, ������ ����������
    sys.exit(app.exec()) #��������� ���� �������

print("TensorFlow ", tf.__version__)
print("GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
application()
