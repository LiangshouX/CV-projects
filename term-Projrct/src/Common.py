"""
通用的文件以及数据操作函数，以及项目的全局配置

"""

import os
import xml.dom.minidom as minidom
import cv2
import csv
import numpy as np
from shutil import copy

class Config:
    """定义项目中相关的配置参数"""
    def __init__(self):
        # HOG 相关的参数
        self.winSize = (20, 20)
        self.blockSize = (8, 8)
        self.blockStride = (4, 4)
        self.cellSize = (8, 8)
        self.nBins = 9
        self.deriveAperture = 1
        self.winSigma = -1.
        self.histogramNormType = 0
        self.nLevels = 64
        self.signeGradient = True

        # 文件路径
        self.projectRoot = (os.path.abspath(os.path.dirname(os.path.dirname(__file__)))).replace("\\", '/')

        self.BCCD_JPEGImages = (os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
                                + '\\BCCD_Dataset\\BCCD\\JPEGImages\\').replace("\\", '/')
        # print(self.BCCD_JPEGImages)
        self.BCCD_Annotations = (os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
                                 + '\\BCCD_Dataset\\BCCD\\Annotations\\').replace("\\", '/')
        self.modelRoot = self.projectRoot + "/model/"
        self.svmModel = self.modelRoot + "svm_model.pkl"

        # BCCD数据集解析后的csv文件
        self.BCCDAnnotationCSV = 'data_set/BCCDAnnotation.csv'

        # 训练集、测试集等存放路径
        self.train_set_image = 'data_set/train_set/Images/'
        self.test_set_image = 'data_set/test_set/Images/'
        self.train_set_annotation = 'data_set/train_set/Annotations/'
        self.test_set_annotation = 'data_set/test_set/Annotations/'
        self.train_label = 'data_set/train_label.txt'
        self.test_label = 'data_set/test_label.txt'
        self.train_hogFeaturePath = 'data_set/train_set/trainHOG.txt'
        self.test_hogFeaturePath = 'data_set/test_set/testHOG.txt'

        # 图像尺寸数量相关
        self.imgSize = (640, 480)   # 原数据集图片的大小
        self.patchSize = (64, 64)   # 训练图片的大小


def filterFiles(directoryPath, extension):
    """
    此函数过滤目录中具有所选扩展名的格式文件,来自BCCD_Dataset提供的 ‘plot.py’文件
            代码地址：https://github.com/Shenggan/BCCD_Dataset
    Args:
        directoryPath (str): relative path of the directory that contains text files
        extension (str): extension file
    Returns:
        The list of filtered files with the selected extension
    """
    relevant_path = directoryPath
    included_extensions = [extension]
    file_names = [file1 for file1 in os.listdir(relevant_path) if
                  any(file1.endswith(ext) for ext in included_extensions)]
    numberOfFiles = len(file_names)
    listParams = [file_names, numberOfFiles]
    return listParams

def build_csv():
    """将BCCD数据集的标注数据写入到CSV文件中"""
    config = Config()
    if not os.path.exists(config.BCCDAnnotationCSV):

        with open(config.BCCDAnnotationCSV, 'w', newline='') as f:
            # 写入表头
            csv_writer = csv.writer(f)
            csv_head = ["filename", "cell_type", "xmin", "ymin", "xmax", "ymax"]
            csv_writer.writerow(csv_head)

            # 读取annotation文件
            [annotation_names, numberOfFiles] = filterFiles(config.BCCD_Annotations, "xml")

            for annotation in annotation_names:
                # 打开xml文档
                dom = minidom.parse(config.BCCD_Annotations + annotation)
                # print(annotation)

                # 得到文件的元素对象
                collection = dom.documentElement

                objects = collection.getElementsByTagName("object")

                file_name = annotation.replace("xml", "jpg")
                # 遍历每个object
                for obj in objects:
                    # 获取标签属性值
                    name = obj.getAttribute("name")
                    box = obj.getAttribute("bndbox")
                    # 获取标签中的内容
                    cell_type = obj.getElementsByTagName('name')[0].childNodes[0].nodeValue
                    boundBox = obj.getElementsByTagName('bndbox')[0]

                    xmin = boundBox.getElementsByTagName('xmin')[0].childNodes[0].nodeValue
                    ymin = boundBox.getElementsByTagName('ymin')[0].childNodes[0].nodeValue
                    xmax = boundBox.getElementsByTagName('xmax')[0].childNodes[0].nodeValue
                    ymax = boundBox.getElementsByTagName('ymax')[0].childNodes[0].nodeValue

                    # 将数据写入到CSV文件当中
                    row = [file_name, cell_type, xmin, ymin, xmax, ymax]
                    print(row)
                    csv_writer.writerow(row)

def crop_single_image():
    """
    从数据集中切分出单独的血细胞图片, 裁剪时先y后x
    """
    config = Config()
    if not os.path.exists(config.train_set_image):
        os.makedirs(config.train_set_image)
    if not os.path.exists(config.train_set_annotation):
        os.makedirs(config.train_set_annotation)
    if not os.path.exists(config.test_set_image):
        os.makedirs(config.test_set_image)
    if not os.path.exists(config.test_set_annotation):
        os.makedirs(config.test_set_annotation)

    num_R, num_W, num_P = 0, 0, 0
    maxNum = 300
    # 读取csv, 按照其中存储的信息索引图片
    with open(config.BCCDAnnotationCSV, 'r') as f:
        csv_reader = csv.reader(f)
        for item in csv_reader:
            # type(item) : list
            # print(item)
            # 忽略第一行
            if csv_reader.line_num == 1:
                continue
            img_path = config.BCCD_JPEGImages + item[0]
            # print(img_path)
            img_type = item[1]
            xmin, ymin, xmax, ymax = eval(item[2]), eval(item[3]), eval(item[4]), eval(item[5])

            img = cv2.imread(img_path)
            img_ = cv2.resize(img[ymin:ymax, xmin:xmax], config.patchSize)

            if img_type == "WBC":
                if num_W >= maxNum:
                    continue
                num_W += 1
                cv2.imwrite(config.train_set_image + 'WBC' + str(num_W) + '.jpg', img_)
            elif img_type == "RBC":
                if num_R >= maxNum * 2:
                    continue
                num_R += 1
                cv2.imwrite(config.train_set_image + 'RBC' + str(num_R) + '.jpg', img_)

            elif img_type == "Platelets":
                if num_P >= maxNum:
                    continue
                num_P += 1
                cv2.imwrite(config.train_set_image + 'Pla' + str(num_P) + '.jpg', img_)
            else:
                print("Error!")
                exit(-2)
            if num_W >= maxNum and num_P >= maxNum and num_R >= maxNum:
                f.close()
                break

            # cv2.imshow("test", img_)
            # cv2.waitKey(0)
            # break

def data_split():
    """将数据切分为训练集测试集"""
    # 从train_set/Images中划分一部分到test_set/Images中
    config = Config()
    [imgNames, numFiles] = filterFiles(config.train_set_image, 'jpg')
    # print(numFiles)

    for i in range(numFiles):
        if i % 4 != 0:
            continue
        copy(src=config.train_set_image+imgNames[i], dst=config.test_set_image+imgNames[i])


def write_img_to_xml(imgPath, xmlPath):
    """对图片进行标注，写入到xml文件中, 参考网络代码
    Args:
        imgPath(str): 图像文件的存放路径
        xmlPath(str): xml文件的存放路径
    Returns:
        None
    """
    # 图像基本信息获取
    img_ = cv2.imread(imgPath)
    imgFolder, imgName = os.path.split(imgPath)
    # print(imgFolder, imgName.split('.')[0][:3])
    h, w, d = img_.shape    # height, width, depth

    # 构建Document对象
    doc = minidom.Document()

    # 标题
    annotation = doc.createElement("Annotation")
    doc.appendChild(annotation)
    # 标注路径
    folder = doc.createElement("Folder")
    folder.appendChild(doc.createTextNode(imgFolder))
    annotation.appendChild(folder)
    # 标注文件名
    filename = doc.createElement("filename")
    filename.appendChild(doc.createTextNode(imgName))
    annotation.appendChild(filename)
    # 标注类别（种类）
    imgType = doc.createElement("Type")
    imgType.appendChild(doc.createTextNode(imgName.split('.')[0][:3]))
    annotation.appendChild(imgType)
    # 标注尺寸
    size = doc.createElement("size")
    annotation.appendChild(size)
    width = doc.createElement("width")
    width.appendChild(doc.createTextNode(str(w)))
    size.appendChild(width)
    height = doc.createElement("height")
    height.appendChild(doc.createTextNode(str(h)))
    size.appendChild(height)
    depth = doc.createElement("depth")
    depth.appendChild(doc.createTextNode(str(d)))
    size.appendChild(depth)

    # 写入文件
    with open(xmlPath, "w") as f:
        doc.writexml(f, indent="\t", addindent="\t", newl="\n", encoding="utf-8")


if __name__ == "__main__":
    """for test and preprocess"""
    build_csv()
    crop_single_image()
    # path = 'data_set/train_set/Images/WBC2.jpg'
    # test_path = './tes.xml'
    # img = cv2.imread(path)
    # f_ = hog_feature(img)
    # print("skimage", len(f_), f_)
    # write_img_to_xml(path, test_path)
    data_split()



