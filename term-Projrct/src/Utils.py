"""
包含图像处理需要使用的一些功能函数：
    图像二值化画函数 convert_to_binary_img、
    图像边缘检测函数 detect_contour、
    HOG特征计算函数 hog_feature
"""

import numpy as np
import cv2
import os
from Common import Config, filterFiles
from skimage import feature as ft

cls_names = ["WBC", "RBC", "Pla"]
cla_labels = {"WBC": 0, "RBC": 1, "Pla": 2}
config = Config()

def convert_to_binary_img_(imgBGR, erode_dilate=True):
    """将图像二值化，以便进行轮廓检测
        参考博客：https://blog.csdn.net/xdg15294969271/article/details/119732470

    cv2.adaptiveThreshold函数说明：(参考自博客：http://t.csdn.cn/1tKsO)
        功能：把图片每个像素点作为中心取N*N的区域，然后计算这个区域的阈值
        src:            需要进行二值化的一张灰度图像
        maxValue:       需要进行二值化的一张灰度图像
        adaptiveMethod: 自适应阈值算法。可选ADAPTIVE_THRESH_MEAN_C 或 ADAPTIVE_THRESH_GAUSSIAN_C
        thresholdType:  opencv提供的二值化方法，只能THRESH_BINARY(0)或者THRESH_BINARY_INV(1)
        blockSize:      要分成的区域大小，上述的N
        C:              常数，每个区域计算出的阈值的基础上在减去这个常数作为这个区域的最终阈值，可以为负数
        dst:            输出图像，可以忽略

    Args:
        imgBGR: source image.
        erode_dilate: 是否进行腐蚀膨胀操作
    Return:
        img_bin: 二值图像

    """
    # 阈值化
    imgBGR = cv2.cvtColor(imgBGR, cv2.COLOR_BGR2GRAY)
    # ret, th1 = cv2.threshold(imgBGR, 127, 255, cv2.THRESH_BINARY)
    th2 = cv2.adaptiveThreshold(imgBGR, 255, cv2.ADAPTIVE_THRESH_MEAN_C, 0, 11, 2)
    # th3 = cv2.adaptiveThreshold(imgBGR, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 0, 11, 2)

    if erode_dilate:
        # 腐蚀膨胀（形态学开运算）
        kernelErosion = np.ones((3, 3), np.uint8)
        kernelDilation = np.ones((3, 3), np.uint8)
        # th1 = cv2.erode(th1, kernelErosion, iterations=2)
        # th1 = cv2.dilate(th1, kernelDilation, iterations=2)

        th2 = cv2.erode(th2, kernelErosion, iterations=2)
        th2 = cv2.dilate(th2, kernelDilation, iterations=2)
        #
        # th3 = cv2.erode(th3, kernelErosion, iterations=2)
        # th3 = cv2.dilate(th3, kernelDilation, iterations=2)
    # cv2.imshow("bin", th2)
    cv2.waitKey(0)
    return th2


def convert_to_binary_img(imgBGR, erode_dilate=True):
    """将图像二值化，以便进行轮廓检测
        HSV色彩空间介绍：https://zh.wikipedia.org/wiki/HSL%E5%92%8CHSV%E8%89%B2%E5%BD%A9%E7%A9%BA%E9%97%B4

        先将RGB空间的图像转换至HSV空间，通过颜色阈值分割选出血细胞的两种颜色：蓝色和红色 所对应的区域得到二值化图像。

        HSV彩色阈值编辑器cv2.inRange函数用法：(参考自：http://t.csdn.cn/iJAQn)
        HSV彩色阈值分割参考博客：http://t.csdn.cn/9Rw1l

    Args:
        imgBGR(np.ndarray): 待处理的图片
        erode_dilate(bool): 是否形态学开运算
    Return:
        img_bin(np.ndarray): 二值化的图像

    """
    rows, cols, _ = imgBGR.shape
    imgHSV = cv2.cvtColor(imgBGR, cv2.COLOR_BGR2HSV)

    # 蓝色的HSV空间阈值
    bMin = np.array([90, 43, 180])
    bMax = np.array([179, 127, 244])
    img_Bbin = cv2.inRange(imgHSV, bMin, bMax)

    # 红色的HSV空间阈值
    rMin1 = np.array([0, 30, 46])
    rMax1 = np.array([90, 255, 228])
    img_Rbin1 = cv2.inRange(imgHSV, rMin1, rMax1)

    # 红色的空间第二种阈值
    rMin2 = np.array([156, 30, 46])
    rMax2 = np.array([180, 255, 228])
    img_Rbin2 = cv2.inRange(imgHSV, rMin2, rMax2)
    img_Rbin = np.maximum(img_Rbin1, img_Rbin2)

    img_bin = np.maximum(img_Bbin, img_Rbin)

    if erode_dilate is True:
        # 腐蚀膨胀（形态学开运算）
        kernelErosion = np.ones((9, 9), np.uint8)
        kernelDilation = np.ones((9, 9), np.uint8)

        img_bin = cv2.erode(img_bin, kernelErosion, iterations=2)
        img_bin = cv2.dilate(img_bin, kernelDilation, iterations=2)

    return img_bin

def LoG(image):
    """LOG算子进行图像滤波"""
    kernel = np.array([[0, 0, 1, 0, 0], [0, 1, 2, 1, 0], [1, 2, -16, 2, 1], [0, 1, 2, 1, 0], [0, 0, 1, 0, 0]], dtype=int)
    # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    img_ = cv2.filter2D(image, cv2.CV_16S, kernel)
    return cv2.convertScaleAbs(img_)


def detect_contour(imgBin, min_area=750, max_area=30720, wh_ratio=2.0):
    """边缘检测
    Args:
        imgBin(np.ndarray): 一幅二值图像
        min_area(int): 能被检测到的最小图像区域
        max_area(int): 能被检测到的最大图像区域
        wh_ratio(float): 大边和短边之间的比例
    Returns:
        rectangles(list):
    """
    rectangles = []
    contours, _ = cv2.findContours(imgBin.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if len(contours) == 0:
        return []
    if max_area < 0:
        max_area = imgBin.shape[0] * imgBin.shape[1]

    # 逐个遍历边缘
    for contour in contours:
        area = cv2.contourArea(contour)
        if min_area <= area <= max_area:
            x, y, w, h = cv2.boundingRect(contour)
            if 1.0 * w / h < wh_ratio and 1.0 * h / w < wh_ratio:
                rectangles.append([x, y, w, h])

    return rectangles


def hog_feature(img_array, resize=(64, 64)):
    """从图像中提取出HOG特征
    """
    img_ = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
    img_ = cv2.resize(img_, (64, 64))
    # hog = cv2.HOGDescriptor(config.winSize, config.blockSize, config.blockStride,
    #                         config.cellSize, config.nBins)
    # feature = hog.compute(img_)

    feature_ = ft.hog(img_, orientations=config.nBins, pixels_per_cell=config.blockSize,
                      cells_per_block=config.blockStride, block_norm="L2", transform_sqrt=True)
    return feature_

def save_data_hog():
    """提取训练集和测试集的HOG特征，并写入对应文件中
    """
    # 训练集
    if not os.path.exists(config.train_hogFeaturePath):
        with open(config.train_hogFeaturePath, 'w', newline='\n') as f:
            # 逐个读取训练图像
            [imgNames, _] = filterFiles(config.train_set_image, 'jpg')
            for imgName in imgNames:
                img_ = cv2.imread(config.train_set_image + imgName)
                hogFeature = hog_feature(img_, config)
                hog_str = ','.join(str(i) for i in hogFeature) + '\n'
                f.write(hog_str)

    # 测试集
    if not os.path.exists(config.test_hogFeaturePath):
        with open(config.test_hogFeaturePath, 'w', newline='\n') as f:
            # 逐个读取训练图像
            [imgNames, _] = filterFiles(config.test_set_image, 'jpg')
            for imgName in imgNames:
                img_ = cv2.imread(config.test_set_image + imgName)
                hogFeature = hog_feature(img_, config)
                hog_str = ','.join(str(i) for i in hogFeature) + '\n'
                f.write(hog_str)

def load_hog_and_label(hogPath, dataPath):
    """加载hog特征"""
    imgNames, labels, hogFeatures = [], [], []

    with open(hogPath, 'r') as f:
        data = f.readlines()
        for row in data:
            hogFeature = row.split(',')
            hogFeature = [float(hog) for hog in hogFeature]
            hogFeatures.append(hogFeature)
            # hogFeature = np.array(hogFeature)
            # row = row.rstrip()
            # print(type(row), row)
            # print(type(hogFeature), hogFeature)
    [names, _] = filterFiles(dataPath, 'jpg')
    imgNames = [name[:3] for name in names]
    labels = [cla_labels[item] for item in imgNames]
    # print(imgNames)
    # print(labels)
    return imgNames, np.array(labels), np.array(hogFeatures)

def HSV_range_test():
    """测试出血细胞颜色的HSV空间"""
    imgBGR = cv2.imread(config.train_set_image + "RBC1.jpg")
    imgHSV = cv2.cvtColor(imgBGR, cv2.COLOR_BGR2HSV)
    print(imgHSV)
    hMin, hMax = np.min(imgHSV[:, :, 0]), np.max(imgHSV[:, :, 0])
    print("H\t", hMin, hMax)

    sMin, sMax = np.min(imgHSV[:, :, 1]), np.max(imgHSV[:, :, 1])
    print("S\t", sMin, sMax)

    vMin, vMax = np.min(imgHSV[:, :, 2]), np.max(imgHSV[:, :, 2])
    print("V\t", vMin, vMax)

    print(imgHSV[100, 60, :])
    # cv2.imshow("orig", imgBGR)
    # cv2.imshow("hsv", imgHSV)
    cv2.waitKey(0)

if __name__ == "__main__":
    """preparatory work"""
    save_data_hog()

    img = cv2.imread(config.BCCD_JPEGImages + "BloodImage_00010.jpg")
    cv2.imshow("color", img)

    imgBin = convert_to_binary_img(img)
    cv2.imshow("bin", imgBin)
    cv2.waitKey(0)

    # HSV_range_test()

