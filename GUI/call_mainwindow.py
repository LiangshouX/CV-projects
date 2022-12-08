# -*- coding: utf-8 -*-
"""
Form implementation generated from reading ui file 'mainwindow.ui'

"""
import os
import random
import sys
import cv2
from PIL import Image
import numpy as np
# PyQt5中使用的基本控件
from PyQt5 import QtGui
from PyQt5.QtGui import QPalette, QImage, QPixmap
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtWidgets import QApplication, QMainWindow, QDialog, QFileDialog
from mainwindow import Ui_MainWindow

def load_label(label_path):
    labels = []
    with open(label_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            labels.append(line.strip("\n"))
    return labels

def cvImgtoQtImg(cvImg):
    """定义opencv图像转PyQt图像的函数"""
    QtImgBuf = cv2.cvtColor(cvImg, cv2.COLOR_BGR2BGRA)
    QtImg = QtGui.QImage(QtImgBuf.data, QtImgBuf.shape[1], QtImgBuf.shape[0], QtGui.QImage.Format_RGB32)
    return QtImg

def getImagesAndLabels(face_detector, path):
    imagePaths = [os.path.join(path, f) for f in os.listdir(path)]
    faceSamples = []
    ids = []

    for imagePath in imagePaths:

        PIL_img = Image.open(imagePath).convert('L')
        img_numpy = np.array(PIL_img, 'uint8')

        id = int(os.path.split(imagePath)[-1].split(".")[1])
        faces = face_detector.detectMultiScale(img_numpy)

        for (x, y, w, h) in faces:
            faceSamples.append(img_numpy[y:y + h, x:x + w])
            ids.append(id)

    return faceSamples, ids


def single_img_recognition(img, faceCascade, face_recognizer, names):
    """对单张人脸图像识别"""
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    font = cv2.FONT_HERSHEY_TRIPLEX
    minW, minH = 64, 48
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=5,
        minSize=(int(minW), int(minH)),
    )

    for (x, y, w, h) in faces:
        # 绘制矩形框
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        id, confidence = face_recognizer.predict(gray[y:y + h, x:x + w])
        # 检查置信度
        if confidence < 100:
            id = names[id]
            confidence = "  {0}%".format(round(100 - confidence))
        else:
            id = "unknown"
            confidence = "  {0}%".format(round(100 - confidence))

        cv2.putText(img, str(id), (x + 5, y - 5), font, 1, (255, 255, 255), 2)
        cv2.putText(img, str(confidence), (x + 5, y + h - 5), font, 1, (255, 255, 0), 1)
    return img

class Config:
    def __init__(self):
        self.data_path_pub = "../dataset/pubfig/"
        self.data_path_usr = "../dataset/User/"
        self.train_path_pub = "../trainer/trainer_pub.yml"
        self.train_path_usr = "../trainer/trainer_user.yml"
        self.batch = 4
        self.cascade = "../models/haarcascade_frontalface_default.xml"
        self.usr_label = "../dataset/user_label.txt"
        self.pub_label = "../dataset/pubfig_label.txt"
        self.video_path = None
        self.gui = True


class MyMainWindow(QMainWindow, Ui_MainWindow):
    def __init__(self, parent=None):
        super(MyMainWindow, self).__init__(parent)
        self.setupUi(self)

        # 配置信息
        self.config = Config()
        self.config.gui = True
        self.commitClicked = False
        self.saved_usr_label = False
        self.bClose = False
        self.captureCount = 0
        self.mode = 0  # 执行对应的功能
        self.fps = 30
        self.names_usr = load_label(self.config.usr_label)
        self.names_pub = load_label(self.config.pub_label)

        # 摄像头
        self.camera = cv2.VideoCapture(0)
        self.is_camera_opened = False  # 摄像头有没有打开标记
        self.face_recognizer = cv2.face.LBPHFaceRecognizer_create()
        self.font = cv2.FONT_HERSHEY_TRIPLEX

        # 定时器：30ms捕获一帧
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._queryFrame)
        self._timer.setInterval(10)

        # 添加槽函数
        self.recordBtn.clicked.connect(self.recordFace)
        self.cameraBtn.clicked.connect(self.openCamera)
        self.selectImgBtn.clicked.connect(self.selectImg)
        self.selectVidBtn.clicked.connect(self.selectVideos)
        self.TrainUser.clicked.connect(self.userTrain)
        self.TrainPub.clicked.connect(self.pubTrain)
        self.pushButton.clicked.connect(self.commitID)

    def recordFace(self):
        """记录用户人脸"""
        self.mode = 0
        # cam = cv2.VideoCapture(0)
        self.camera.set(3, 800)
        self.camera.set(4, 500)
        self.is_camera_opened = ~self.is_camera_opened

        if self.is_camera_opened:
            if self.commitClicked:
                self.commitClicked = False
                self.recordBtn.setText("Stop Record")
                self._timer.start()
        else:
            self.saved_usr_label = False
            self.recordBtn.setText("Record Face")
            self._timer.stop()
            # self.camera.release()

    def openCamera(self):
        """打开摄像头实时人脸识别"""
        self.showLabel.clear()
        self.mode = 1
        # self.camera.read()
        # self.camera = cv2.VideoCapture(0)
        self.camera.set(3, 800)
        self.camera.set(4, 500)
        self.is_camera_opened = ~self.is_camera_opened

        faceCascade = cv2.CascadeClassifier(self.config.cascade)

        # -------------------------------算法部分----------------------------#
        self.face_recognizer.read(self.config.train_path_usr)
        while not self.bClose:
            ret, img = self.camera.read()
            if not ret:
                self.msgBrowser.setText("Error!")
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            minW, minH = 64, 48
            # names = ['None'] + load_label(self.config.pub_label)

            faces = faceCascade.detectMultiScale(
                gray,
                scaleFactor=1.2,
                minNeighbors=5,
                minSize=(int(minW), int(minH)),
            )

            for (x, y, w, h) in faces:
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                id, confidence = self.face_recognizer.predict(gray[y:y + h, x:x + w])
                # 检查置信度
                if confidence < 100:
                    id = self.names_usr[id]
                    confidence = "  {0}%".format(round(100 - confidence))
                else:
                    id = "unknown"
                    confidence = "  {0}%".format(round(100 - confidence))

                cv2.putText(img, str(id), (x + 5, y - 5), self.font, 1, (255, 255, 255), 2)
                cv2.putText(img, str(confidence), (x + 5, y + h - 5), self.font, 1, (255, 255, 0), 1)

            # 在mainWindow中显示
            QtImg = cvImgtoQtImg(img)  # 单帧图像转换为PyQt图像格式
            self.showLabel.setPixmap(QtGui.QPixmap.fromImage(QtImg))
            self.showLabel.show()  # 刷新界面
            cv2.waitKey(int(500 / self.fps))

    def selectImg(self):
        """从本地选择图片作为识别对象"""
        self.mode = 2
        self.face_recognizer.read(self.config.train_path_pub)
        # 级联分类器
        faceCascade = cv2.CascadeClassifier(self.config.cascade)
        # 定义被识别为人脸的最小窗口大小
        minW = 64  # 宽度
        minH = 48  # 高度

        # 显示的宽度和高度
        imgH = 800
        imgW = 450

        # imagePaths = [os.path.join(self.config.data_path_pub, f) for f in os.listdir(self.config.train_path_pub)]
        # count = 1
        # batch_list = []
        # for imagepath in imagePaths:
        #     img = cv2.imread(imagepath)
        #     img = single_img_recognition(img, faceCascade, self.face_recognizer, self.names_pub)
        #     cv2.resize(img, (imgW, imgH))
        #
        #     if count < self.config.batch:
        #         count += 1
        #         batch_list.append(img)
        #     else:
        #         merge_img = np.hstack(batch_list)
        #         merge_img = cvImgtoQtImg(merge_img)
        #         # self.showLabel.setText("Loaded")
        #         self.showLabel.setPixmap(QtGui.QPixmap.fromImage(merge_img))
        #         cv2.waitKey(10000)
        #         count = 0
        #         batch_list.clear()

        filepath, _ = QFileDialog.getOpenFileName(self, '打开图片')
        img = cv2.imread(filepath)
        self.msgBrowser.setText("Load Successful!")
        img = single_img_recognition(img, faceCascade, self.face_recognizer, self.names_pub)
        # img = cv2.resize(img, (imgW, imgH))

        img = cvImgtoQtImg(img)
        self.showLabel.setPixmap(QtGui.QPixmap.fromImage(img))
        self.showLabel.show()  # 刷新界面
        cv2.waitKey(30)

    def selectVideos(self):
        """选择视频作为识别对象"""
        self.mode = 3
        self.face_recognizer.read(self.config.train_path_pub)

        # 打开文件选取对话框
        self.config.video_path, _ = QFileDialog.getOpenFileName(self, '打开视频')

        cam = cv2.VideoCapture(self.config.video_path)
        fps = cam.get(cv2.CAP_PROP_FPS)     # 视频的帧率

        if not cam.isOpened():
            self.msgBrowser.setText("视频文件打开失败！")
            exit(-2)

        faceCascade = cv2.CascadeClassifier(self.config.cascade)
        while not self.bClose:
            ret, img = cam.read()
            if not ret:
                if img is None:
                    self.msgBrowser.setText("视频播放结束！")
                else:
                    self.msgBrowser.setText("视频播放错误！")
                break
            # -------------------------------------------识别算法部分-----------------------------#
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            minW, minH = 64, 48
            # names = ['None'] + load_label(self.config.pub_label)

            faces = faceCascade.detectMultiScale(
                gray,
                scaleFactor=1.2,
                minNeighbors=5,
                minSize=(int(minW), int(minH)),
            )

            for (x, y, w, h) in faces:
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                id, confidence = self.face_recognizer.predict(gray[y:y + h, x:x + w])
                # 检查置信度
                if confidence < 100:
                    id = self.names_pub[id]
                    confidence = "  {0}%".format(round(100 - confidence))
                else:
                    id = "unknown"
                    confidence = "  {0}%".format(round(100 - confidence))

                cv2.putText(img, str(id), (x + 5, y - 5), self.font, 1, (255, 255, 255), 2)
                cv2.putText(img, str(confidence), (x + 5, y + h - 5), self.font, 1, (255, 255, 0), 1)
            # ----------------------------------------------------------------------------------------------#

            # 在mainWindow中显示
            QtImg = cvImgtoQtImg(img)   # 单帧图像转换为PyQt图像格式
            self.showLabel.setPixmap(QtGui.QPixmap.fromImage(QtImg))
            self.showLabel.show()   # 刷新界面
            cv2.waitKey(int(500 / fps))

    def userTrain(self):
        """训练用户数据"""
        path = self.config.data_path_usr
        face_detector = cv2.CascadeClassifier(self.config.cascade)
        # face_recognizer = cv2.face.LBPHFaceRecognizer_create()
        self.msgBrowser.setText("[INFO] Training faces. It will take a few seconds. Wait ...")
        faces, ids = getImagesAndLabels(face_detector, path)
        self.face_recognizer.train(faces, np.array(ids))
        # 将模型保存至 trainer/trainer_user.yml 中
        self.face_recognizer.write(self.config.train_path_usr)
        self.msgBrowser.setText("[INFO] {0} faces trained. Exiting Program".format(len(np.unique(ids))))

    def pubTrain(self):
        """训练pubfig数据集"""
        path = self.config.data_path_pub
        face_detector = cv2.CascadeClassifier(self.config.cascade)
        # face_recognizer = cv2.face.LBPHFaceRecognizer_create()
        self.msgBrowser.setText("[INFO] Training faces. It will take a few seconds. Wait ...")
        faces, ids = getImagesAndLabels(face_detector, path)
        self.face_recognizer.train(faces, np.array(ids))
        # 将模型保存至 trainer/trainer_pub.yml 中
        self.face_recognizer.write(self.config.train_path_pub)
        self.msgBrowser.setText("[INFO] {0} faces trained. Exiting Program".format(len(np.unique(ids))))

    def commitID(self):
        # id, name = self.idEdit.text(), self.nameEdit.text()
        self.commitClicked = True

    def _queryFrame(self):
        """
        循环捕获图片
        """
        ret, self.frame = self.camera.read()

        # -------------------------------------用户人脸采集部分--------------------------------------#
        if self.mode == 0:
            count = 0
            face_detector = cv2.CascadeClassifier(self.config.cascade)
            # recognizer = cv2.face.LBPHFaceRecognizer_create()
            face_id = self.idEdit.text()
            user_name = self.nameEdit.text()

            gray = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)
            faces = face_detector.detectMultiScale(gray, 1.3, 5)

            for (x, y, w, h) in faces:
                cv2.rectangle(self.frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                # count += 1
                self.captureCount += 1
                # 将采集到的人脸写入本地数据集中
                if not os.path.exists(self.config.data_path_usr):
                    os.mkdir(self.config.data_path_usr)
                cv2.imwrite(
                    self.config.data_path_usr + "User." + str(face_id) + '.' + str(self.captureCount) + ".jpg",
                    gray[y:y + h, x:x + w])
            self.msgBrowser.setText("Capturing faces :" + str(self.captureCount))

            if not self.saved_usr_label:
                with open(self.config.usr_label, 'a+') as f:
                    f.write("\n" + user_name)
                self.saved_usr_label = True
        # --------------------------------------------------------------------------------------#

        # -------------------------------------打开摄像头进行用户人脸识别-----------------------------#
        # -----------------------------------------停止用户人脸识别---------------------------------#

        # --------------------------------------选择图片进行检测-------------------------------------#
        # -----------------------------------------------------------------------------------------#

        # --------------------------------------选择视频进行检测-------------------------------------#
        # -----------------------------------------------------------------------------------------#

        img_rows, img_cols, channels = self.frame.shape
        bytesPerLine = channels * img_cols

        # QImg = cvImgtoQtImg(self.frame)
        cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB, self.frame)
        QImg = QImage(self.frame.data, img_cols, img_rows, bytesPerLine, QImage.Format_RGB888)
        self.showLabel.setPixmap(QPixmap.fromImage(QImg).scaled(
            self.showLabel.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
        self.showLabel.setPixmap(QtGui.QPixmap.fromImage(QImg))
        # cv2.waitKey(int(500 / 30))


if __name__ == "__main__":
    # 固定格式，PyQt5程序都需要QApplication对象。sys.argv是命令行参数列表，确保程序可以双击运行
    app = QApplication(sys.argv)
    # 初始化
    myWin = MyMainWindow()

    pe = QPalette()
    pe.setColor(QPalette.Window, Qt.white)
    myWin.showLabel.setAutoFillBackground(True)  # 设置背景充满，为设置背景颜色的必要条件
    myWin.showLabel.setPalette(pe)
    # 将窗口控件显示在屏幕上
    myWin.show()

    sys.exit(app.exec_())


