# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'reg.ui'
##
## Created by: Qt User Interface Compiler version 6.7.0
##
## WARNING! All changes made in this file will be lost when recompiling UI file!
################################################################################

from PySide6.QtCore import (QCoreApplication, QDate, QDateTime, QLocale,
    QMetaObject, QObject, QPoint, QRect,
    QSize, QTime, QUrl, Qt)
from PySide6.QtGui import (QBrush, QColor, QConicalGradient, QCursor,
    QFont, QFontDatabase, QGradient, QIcon,
    QImage, QKeySequence, QLinearGradient, QPainter,
    QPalette, QPixmap, QRadialGradient, QTransform)
from PySide6.QtWidgets import (QApplication, QLabel, QPlainTextEdit, QPushButton,
    QSizePolicy, QWidget)

class Ui_RegisterWin(object):
    def setupUi(self, RegisterWin):
        if not RegisterWin.objectName():
            RegisterWin.setObjectName(u"RegisterWin")
        RegisterWin.resize(343, 258)
        self.label = QLabel(RegisterWin)
        self.label.setObjectName(u"label")
        self.label.setGeometry(QRect(10, 20, 261, 16))
        self.sel_lic_file = QPushButton(RegisterWin)
        self.sel_lic_file.setObjectName(u"sel_lic_file")
        self.sel_lic_file.setGeometry(QRect(30, 160, 261, 31))
        self.activat_but = QPushButton(RegisterWin)
        self.activat_but.setObjectName(u"activat_but")
        self.activat_but.setGeometry(QRect(130, 220, 81, 31))
        self.lic_code_txt = QPlainTextEdit(RegisterWin)
        self.lic_code_txt.setObjectName(u"lic_code_txt")
        self.lic_code_txt.setGeometry(QRect(10, 40, 321, 61))
        self.label_2 = QLabel(RegisterWin)
        self.label_2.setObjectName(u"label_2")
        self.label_2.setGeometry(QRect(10, 140, 141, 16))

        self.retranslateUi(RegisterWin)

        QMetaObject.connectSlotsByName(RegisterWin)
    # setupUi

    def retranslateUi(self, RegisterWin):
        RegisterWin.setWindowTitle(QCoreApplication.translate("RegisterWin", u"Form", None))
        self.label.setText(QCoreApplication.translate("RegisterWin", u"Past your licence here", None))
        self.sel_lic_file.setText(QCoreApplication.translate("RegisterWin", u"Select Licence File", None))
        self.activat_but.setText(QCoreApplication.translate("RegisterWin", u"Activate", None))
        self.lic_code_txt.setPlainText("")
        self.label_2.setText(QCoreApplication.translate("RegisterWin", u"Or set licence file", None))
    # retranslateUi

