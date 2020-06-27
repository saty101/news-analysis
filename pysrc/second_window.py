# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'ps1.2.ui'
#
# Created by: PyQt5 UI code generator 5.9.2
#
# WARNING! All changes made in this file will be lost!
import traceback

from sentiment_analyser import classification, textsum, savalue
from PyQt5 import QtCore, QtGui, QtWidgets

with open('ARTICLE.txt', 'r') as File:
    news_text = File.read()


class Ui_MainWindow(object):

    def __init__(self, cb1, cb2, cb3):
        self.centralwidget = None
        self.textBrowser = None
        self.cb1_check = cb1
        self.cb2_check = cb2
        self.cb3_check = cb3

    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(773, 481)
        MainWindow.setContextMenuPolicy(QtCore.Qt.DefaultContextMenu)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.textBrowser = QtWidgets.QTextBrowser(self.centralwidget)
        self.textBrowser.setGeometry(QtCore.QRect(60, 50, 351, 311))
        self.textBrowser.setObjectName("textBrowser")

        self.textBrowser_2 = QtWidgets.QTextBrowser(self.centralwidget)
        self.textBrowser_2.setGeometry(QtCore.QRect(460, 40, 256, 31))
        self.textBrowser_2.setObjectName("textBrowser_2")

        self.textBrowser_3 = QtWidgets.QTextBrowser(self.centralwidget)
        self.textBrowser_3.setGeometry(QtCore.QRect(460, 240, 256, 192))
        self.textBrowser_3.setObjectName("textBrowser_3")

        self.textBrowser_4 = QtWidgets.QTextBrowser(self.centralwidget)
        self.textBrowser_4.setGeometry(QtCore.QRect(460, 110, 256, 91))
        self.textBrowser_4.setObjectName("textBrowser_4")

        font = QtGui.QFont()
        font.setPointSize(7)
        self.pushButton_2 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_2.setGeometry(QtCore.QRect(140, 380, 131, 23))
        font = QtGui.QFont()
        font.setPointSize(7)
        self.pushButton_2.setFont(font)
        self.pushButton_2.setObjectName("pushButton_2")
        font = QtGui.QFont()
        font.setPointSize(7)
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(170, 10, 151, 21))
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(10)
        self.label.setFont(font)
        self.label.setObjectName("label")
        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        self.label_2.setGeometry(QtCore.QRect(500, 10, 171, 20))
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.label_2.setFont(font)
        self.label_2.setObjectName("label_2")
        self.label_3 = QtWidgets.QLabel(self.centralwidget)
        self.label_3.setGeometry(QtCore.QRect(470, 80, 231, 20))
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.label_3.setFont(font)
        self.label_3.setObjectName("label_3")
        self.label_4 = QtWidgets.QLabel(self.centralwidget)
        self.label_4.setGeometry(QtCore.QRect(500, 210, 161, 20))
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.label_4.setFont(font)
        self.label_4.setObjectName("label_4")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 773, 22))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

        if self.cb1_check == True:
            try:
                x = list()
                x = classification(news_text)
                for i in x:
                    self.textBrowser_2.setText(i)
            except Exception as exc:
                traceback.print_exc()

        else:
            self.textBrowser_2.setText("")

        if self.cb2_check == True:
            try:
                w,v = savalue(news_text)
                senti = "Predicted Sentiment Value "+w + "\nPredicted Sentiment "+v
                self.textBrowser_4.setText(senti)
            except Exception as exc:
                traceback.print_exc()

        else:
            self.textBrowser_4.setText("")

        if self.cb3_check == True:
            try:
                y = list()
                y = textsum(news_text)
                for j in y:
                    self.textBrowser_3.setText(j)
            except Exception as exc:
                traceback.print_exc()

        else:
            self.textBrowser_3.setText("")


        self.textBrowser.setText(news_text)






    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "Sentiment Analysis OUTPUT"))
        self.pushButton_2.setText(_translate("MainWindow", "NEXT ITERATION"))
        self.label.setText(_translate("MainWindow", "<html><head/><body><p align=\"center\"><span style=\" font-weight:600;\">NEWS ARTICLE</span></p></body></html>"))
        self.label_2.setText(_translate("MainWindow", "<html><head/><body><p align=\"center\">NEWS CATEGORY</p></body></html>"))
        self.label_3.setText(_translate("MainWindow", "<html><head/><body><p align=\"center\">SENTIMENT ANALYSIS</p></body></html>"))
        self.label_4.setText(_translate("MainWindow", "<html><head/><body><p align=\"center\">TEXT SUMMARY</p></body></html>"))


def start_window(CB1, CB2, CB3):
    try:
        window = QtWidgets.QMainWindow()
        ui = Ui_MainWindow(CB1, CB2, CB3)
        ui.setupUi(window)

        print("starting second window")
        window.show()
        print(window.__dir__())
        print("second window shown")
        print(CB1, CB2, CB3)
    except Exception as exc:
        traceback.print_exc()

if __name__ == "__main__":
    start_app()
