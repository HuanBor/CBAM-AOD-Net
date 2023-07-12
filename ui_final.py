from PyQt5.QtWidgets import QMainWindow, QApplication, QDesktopWidget, QLabel, QFileDialog, QPushButton, QMessageBox, \
    QWidget
import sys
from PyQt5 import QtGui
import cv2
from test import test_on_img_
from darkchannel import test_on_img_dark
from retinex import test_on_img_retinex
import numpy as np
import torchvision.transforms as transforms


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.ui_basic = Ui_basic()
        self.ui_basic.setParent(self)
        self.ui_basic.initUI()

        self.resize(1200, 700)
        self.center()
        self.setWindowTitle('Dehaze')
        self.show()

    def center(self):
        screen = QDesktopWidget().screenGeometry()
        size = self.geometry()
        self.move((screen.width() - size.width()) / 2,
                  (screen.height() - size.height()) / 2)


class Ui_basic(QWidget):
    def __init__(self):
        super().__init__()
        self.src_img = None

    def initUI(self):
        self.resize(1200, 700)

        self.open_src_button = QPushButton(parent=self)
        self.open_src_button.setText("打开图片")
        self.open_src_button.move(200, 20)
        self.open_src_button.pressed.connect(self.open_src_img)

        #AOD-Net buttopn
        self.dehaze_button = QPushButton(parent=self)
        self.dehaze_button.setText("AOD去雾")
        self.dehaze_button.move(self.open_src_button.x() + self.open_src_button.width() + 30, self.open_src_button.y())
        self.dehaze_button.pressed.connect(self.dehze4)

        # darkchannel buttopn
        self.dark_button = QPushButton(parent=self)
        self.dark_button.setText("Darkchannel去雾")
        self.dark_button.move(self.dehaze_button.x() + self.dehaze_button.width() + 30, self.open_src_button.y())
        self.dark_button.pressed.connect(self.dehze2)

        # retinex buttopn
        self.retinex_button = QPushButton(parent=self)
        self.retinex_button.setText("Retinex去雾")
        self.retinex_button.move(self.dark_button.x() + self.dark_button.width() + 60, self.open_src_button.y())
        self.retinex_button.pressed.connect(self.dehze3)

        self.src_img_area = QLabel(parent=self)  # 图形显示区域
        self.src_img_area.resize(500, 500)
        self.src_img_area.move(40, self.open_src_button.y() + self.open_src_button.height() + 20)

        self.result_img_area = QLabel(parent=self)  # 结果图形显示区域
        self.result_img_area.resize(500, 500)
        self.result_img_area.move(self.src_img_area.x() + self.src_img_area.width() + 40, self.src_img_area.y())

    def open_src_img(self):
        fileName, filetype = QFileDialog.getOpenFileName(self,
                                                         "选取文件",
                                                         "./",
                                                         "photo(*.jpg *.png *.bmp);;All Files (*)")

        self.src_img = cv2.imread(fileName)

        try:
            self.src_img.shape
        except:
            QMessageBox.warning(self, '提示', '图片无法打开', QMessageBox.Yes)
            return

        self.showImage(self.src_img_area, self.src_img)

    def showImage(self, qlabel, img):
        size = (int(qlabel.width()), int(qlabel.height()))
        shrink = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
        # cv2.imshow('img', shrink)
        shrink = cv2.cvtColor(shrink, cv2.COLOR_BGR2RGB)
        self.QtImg = QtGui.QImage(shrink.data,
                                  shrink.shape[1],
                                  shrink.shape[0],
                                  QtGui.QImage.Format_RGB888)

        qlabel.setPixmap(QtGui.QPixmap.fromImage(self.QtImg))

    def dehze(self):
        try:
            self.src_img.shape
        except:
            QMessageBox.warning(self, '提示', '请先打开原图片', QMessageBox.Yes)
            return

        result = test_on_img_('Epoch9.pth', self.src_img)
        image = result.cpu().clone()
        image = image.squeeze(0)
        image = transforms.ToPILImage()(image)
        image = np.array(image)

        self.showImage(self.result_img_area, image)


    #暗通道去雾代码调用
    def dehze2(self):
        try:
            self.src_img.shape
        except:
            QMessageBox.warning(self, '提示', '请先打开原图片', QMessageBox.Yes)
            return
        result = test_on_img_dark(self.src_img)
        #result = cv2.imread('J.png')

        self.showImage(self.result_img_area,result)

    # retinex去雾代码调用
    def dehze3(self):
        try:
            self.src_img.shape
        except:
            QMessageBox.warning(self, '提示', '请先打开原图片', QMessageBox.Yes)
            return

        result = test_on_img_retinex(self.src_img)

        self.showImage(self.result_img_area,result)

    def dehze4(self):
        try:
            self.src_img.shape
        except:
            QMessageBox.warning(self, '提示', '请先打开原图片', QMessageBox.Yes)
            return

        result = cv2.imread('AOD.jpg')

        self.showImage(self.result_img_area,result)

if __name__ == "__main__":
    app = QApplication([])
    ui = MainWindow()
    sys.exit(app.exec_())
