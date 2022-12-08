"""
人脸识别的功能类
"""
import cv2
import numpy as np
import os
from cv2 import dnn_superres
from PIL import Image

import argparse

def set_Config():
    parser = argparse.ArgumentParser(description='Face Recognition')
    parser.add_argument('--gui', default=False, type=bool, required=False)
    parser.add_argument('--data_path', default="dataset/pubfig/", type=str, required=False, help="Path of dataset")
    parser.add_argument('--train_path', default="trainer/trainer_user.yml", type=str, required=False,
                        help='choose the train path')
    parser.add_argument('--capture', default=False, type=bool, help='need to capture users face?')
    parser.add_argument('--trained', default=False, type=bool, help='have trained?')
    parser.add_argument('--batch', default=8, type=int, help='batch size to show images')
    parser.add_argument('--cascade', default='E:/ProgrammingFiles/Python/CV/Term-Pro-OpenCV-Face-Recognition'
                                             '/FacialRecognition/models/haarcascade_frontalface_default.xml', type=str,
                        help="Cascade path")

    args = parser.parse_args()
    return args

def load_label(label_path):
    labels = []
    with open(label_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            labels.append(line.strip("\n"))
    return labels

def img_sr(img_path, algorithm='bilinear', scale=4):
    """超分辨率重建
    Args:
        img_path(str): cv2.imread读入的图片格式
        algorithm(str):  超分辨率重建的算法，可选项有：bilinear, bicubic, edsr, fsrcnn
        scale(int)      : 放大的比例
    Returns:
        img_new(numpy.array)
    """
    img_new = None

    # 可选择算法，bilinear, bicubic, edsr, fsrcnn
    # algorithm = "bilinear"

    # 放大比例，可输入值2，3，4
    scale = 4
    # 模型路径
    path = "models/EDSR_x4.pb"

    # 载入图像
    img = cv2.imread(img_path)
    # 如果输入的图像为空
    if img is None:
        print("Couldn't load image: " + str(img_path))
        return

    # 创建模型
    sr = dnn_superres.DnnSuperResImpl_create()

    if algorithm == "bilinear":
        img_new = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
    elif algorithm == "bicubic":
        img_new = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    elif algorithm == "edsr" or algorithm == "fsrcnn":
        # 读取模型
        sr.readModel(path)
        #  设定算法和放大比例
        sr.setModel(algorithm, scale)
        # 放大图像
        img_new = sr.upsample(img)
    else:
        print("Algorithm not recognized")

    # 如果失败
    if img_new is None:
        print("Upsampling failed")

    print("Upsampling succeeded. \n")

    # # 展示图片
    # cv2.namedWindow("Initial Image", cv2.WINDOW_AUTOSIZE)
    # # 初始化图片
    # cv2.imshow("Initial Image", img_new)
    # cv2.imwrite("./saved.jpg", img_new)
    # cv2.waitKey(0)
    return img_new


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
    cv2.imshow('camera', img)


# class FaceRec:
#     def __init__(self, device_conf=0, train_path="dataset/pubfig", capture=False, trained=True):
#         self.capture = capture
#         self.device_conf = device_conf
#         # self.cam = cv2.VideoCapture(device_conf)
#         # self.cam.set(3, 640)  # set video width
#         # self.cam.set(4, 480)  # set video height
#         self.train_path = train_path
#         # self.name_list = []
#
#         # load the face detector
#         self.face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
#
#         self.recognizer = cv2.face.LBPHFaceRecognizer_create()

def user_face_capture(config, face_id=None, user_name=None, cam=None):
    """从多个用户捕获多个face以存储在数据库(dataset目录)中
    """
    # 从摄像头采集数据
    # 一位user对应一个 face id
    if not config.gui:
        cam = cv2.VideoCapture(0)
    cam.set(3, 640)
    cam.set(3, 640)
    face_detector = cv2.CascadeClassifier(config.cascade)
    recognizer = cv2.face.LBPHFaceRecognizer_create()

    if not config.gui:
        face_id = input('\n Enter user id end press <return> ==>  ')
        user_name = input('\n Enter user name end press <return> ==>  ')
    count = 0
    while True:
        ret, img = cam.read()
        # 转换为灰度图像
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_detector.detectMultiScale(gray, 1.3, 5)
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
            count += 1
            # 将采集到的人脸写入本地数据集中
            if not os.path.exists("./dataset/User/user" + str(face_id)):
                os.mkdir("./dataset/User/user" + str(face_id))
            cv2.imwrite(
                "dataset/User/user" + str(face_id) + '/' + "User." + str(face_id) + '.' + str(count) + ".jpg",
                gray[y:y + h, x:x + w])

        cv2.imshow('image', img)

        k = cv2.waitKey(100) & 0xff  # 按 'ESC' 退出采集
        if k == 27:
            # name = input("What is the users name?")
            with open("dataset/user_label.txt", 'a+') as f:
                f.write(user_name)
            break
        elif count >= 50:  # 采集到30张人脸图像
            # name = input("What is the users name?")
            with open("dataset/user_label.txt", 'a+') as f:
                f.write('\n'+user_name)
            break

def face_train(config):
    """训练人脸数据"""
    path = config.data_path
    face_detector = cv2.CascadeClassifier('models/haarcascade_frontalface_default.xml')
    face_recognizer = cv2.face.LBPHFaceRecognizer_create()

    print("\n [INFO] Training faces. It will take a few seconds. Wait ...")
    faces, ids = getImagesAndLabels(face_detector, path)
    face_recognizer.train(faces, np.array(ids))
    # 将模型保存至 trainer/trainer_pub.yml 中
    face_recognizer.write(config.train_path)

    # 输出训练过的人脸数量并结束程序
    print("\n [INFO] {0} faces trained. Exiting Program".format(len(np.unique(ids))))

def face_recognition_camera(config):
    """使用摄像头时，加载用户训练模型"""
    face_recognizer = cv2.face.LBPHFaceRecognizer_create()
    face_recognizer.read(config.train_path)
    cascadePath = "haarcascade_frontalface_default.xml"
    faceCascade = cv2.CascadeClassifier(config.cascade)

    # 设置显示字体的size
    font = cv2.FONT_HERSHEY_SIMPLEX

    id = 0

    # user name 和 ID相关联
    names1 = load_label("dataset/user_label.txt")
    names2 = load_label("dataset/pubfig_label.txt")
    names = names1
    print(names)

    cam = cv2.VideoCapture(0)
    cam.set(3, 640)  # set video widht
    cam.set(4, 480)  # set video height

    # 定义被识别为人脸的最小窗口大小
    minW = 0.1 * cam.get(3)
    minH = 0.1 * cam.get(4)

    while True:
        ret, img = cam.read()
        single_img_recognition(img, faceCascade, face_recognizer, names)

        k = cv2.waitKey(10) & 0xff  # Press 'ESC' for exiting video
        if k == 27:
            break

    print("\n [INFO] Exiting Program and cleanup stuff")
    cam.release()
    cv2.destroyAllWindows()

def face_recognition_image(config):
    # cam = cv2.VideoCapture()

    face_recognizer = cv2.face.LBPHFaceRecognizer_create()
    face_recognizer.read(config.train_path)
    # 级联分类器
    faceCascade = cv2.CascadeClassifier(config.cascade)

    font = cv2.FONT_HERSHEY_SIMPLEX

    id = 0

    names = load_label("dataset/pubfig_label.txt")

    # 定义被识别为人脸的最小窗口大小
    minW = 64  # 宽度
    minH = 48  # 高度

    # 显示的宽度和高度
    imgH = 200
    imgW = 256

    dataset_path = "dataset/pubfig"
    imagePaths = [os.path.join(dataset_path, f) for f in os.listdir(dataset_path)]
    for imagepath in imagePaths:
        img = cv2.imread(imagepath)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = faceCascade.detectMultiScale(
            gray,
            scaleFactor=1.2,
            minNeighbors=5,
            minSize=(int(minW), int(minH)),
        )

        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            id, confidence = face_recognizer.predict(gray[y:y + h, x:x + w])

            if confidence < 100:
                id = names[id]
                confidence = "  {0}%".format(round(100 - confidence))
            else:
                id = "unknown"
                confidence = "  {0}%".format(round(100 - confidence))

            cv2.putText(img, str(id), (x + 5, y - 5), font, 1, (255, 255, 255), 2)
            cv2.putText(img, str(confidence), (x + 5, y + h - 5), font, 1, (255, 255, 0), 1)

        img_show = cv2.resize(img, (imgW, imgH))
        cv2.imshow('camera', img_show)

        cv2.waitKey(100)

    print("\n [INFO] Exiting Program and cleanup stuff")

    cv2.destroyAllWindows()

def face_recognition_video(config, video_path="speech.mp4"):
    face_recognizer = cv2.face.LBPHFaceRecognizer_create()
    face_recognizer.read(config.train_path)

    cascadePath = "haarcascade_frontalface_default.xml"
    faceCascade = cv2.CascadeClassifier(config.cascade)

    font = cv2.FONT_HERSHEY_TRIPLEX

    # iniciate id counter
    id = 0

    # names related to ids: example ==> Marcelo: id=1,  etc
    names1 = load_label("dataset/user_label.txt")
    names2 = load_label("dataset/pubfig_label.txt")
    names =  names2
    print(names)

    # Define min window size to be recognized as a face
    minW = 64
    minH = 48

    cam = cv2.VideoCapture(video_path)
    while True:
        ret, img = cam.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = faceCascade.detectMultiScale(
            gray,
            scaleFactor=1.2,
            minNeighbors=5,
            minSize=(int(minW), int(minH)),
        )

        for (x, y, w, h) in faces:
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

        cv2.imshow('camera', img)

        k = cv2.waitKey(10) & 0xff  # Press 'ESC' for exiting video
        if k == 27:
            break

    print("\n [INFO] Exiting Program and cleanup stuff")
    cam.release()
    cv2.destroyAllWindows()

def getImagesAndLabels(face_detector, path):
    imagePaths = [os.path.join(path, f) for f in os.listdir(path)]
    faceSamples = []
    ids = []

    for imagePath in imagePaths:

        PIL_img = Image.open(imagePath).convert('L')  # convert it to grayscale
        img_numpy = np.array(PIL_img, 'uint8')

        # img = img_sr(imagePath)
        # img_numpy = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        # print(img_numpy.shape)

        id = int(os.path.split(imagePath)[-1].split(".")[1])
        faces = face_detector.detectMultiScale(img_numpy)

        for (x, y, w, h) in faces:
            faceSamples.append(img_numpy[y:y + h, x:x + w])
            ids.append(id)

    return faceSamples, ids


if __name__ == "__main__":
    args = set_Config()

    if args.data_path == "dataset/pubfig/":
        if not args.trained:
            # 未经过训练
            face_train(args)
        face_recognition_video(args)
        face_recognition_image(args)

    elif args.data_path == "dataset/User/":
        if not args.trained:
            face_train(args)
        decision = input("Record a new face? y or n:")
        if decision == 'y':
            user_face_capture(args)
        else:
            face_recognition_camera(args)


