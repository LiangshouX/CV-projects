"""
    SVM分类器

"""

from Common import Config, filterFiles
from Utils import convert_to_binary_img, detect_contour, hog_feature, save_data_hog, load_hog_and_label

from sklearn.svm import SVC
from sklearn.metrics import roc_curve

import joblib
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

cls_names = ["WBC", "RBC", "Pla"]
cla_labels = {0: "WBC", 1: "RBC", 2: "Pla"}
config = Config()

class SVMCls:
    """SVM分类器
    Args:
        hogFeatures(np.array):  训练数据
        labels(np.array):       训练标签
        test_hogFeatures(np.array):测试数据
        test_labels(np.array):  测试标签
    """
    def __init__(self, hogFeatures, labels, test_hogFeatures, test_labels):
        self.hogFeatures = hogFeatures
        self.labels = labels

        self.test_higFeatures = test_hogFeatures
        self.test_labels = test_labels

    def train(self):
        """训练分类器，并保存至指定路径"""
        print("Start Training, waiting...")
        classifier = SVC(C=10, tol=1e-5, probability=True)
        classifier.fit(self.hogFeatures, self.labels)
        joblib.dump(classifier, config.modelRoot + "svm_model.pkl")

        print("Train Finished! ")

    def test(self):
        """测试分类器在测试数据集上的性能"""
        classifier = joblib.load(config.modelRoot + "svm_model.pkl")
        accuracy = classifier.score(self.test_higFeatures, self.test_labels)
        # classifier.predict_prob
        return accuracy

    def evaluate_model(self):
        classifier = joblib.load(config.modelRoot + "svm_model.pkl")
        predict_labels = []
        for i in range(self.test_labels.shape[0]):
            feature = np.reshape(self.test_higFeatures[i], (1, -1))
            predict_prob = classifier.predict_proba(feature)
            predict_label = np.argmax(predict_prob)
            predict_labels.append(predict_label)
        predict_labels = np.array(predict_labels)
        acc = []
        score_record = 0
        for i in range(predict_labels.shape[0]):
            if predict_labels[i] == self.test_labels[i]:
                score_record += 1
                acc.append(score_record / (i + 1))

        x = np.arange(0, self.test_labels.shape[0])
        plt.figure(num=1)
        plt.xlabel('Step')
        plt.ylabel('Accuracy')
        plt.title('SVM Model')
        plt.plot(x, acc)
        plt.show()
        plt.savefig(config.modelRoot + "model.jpg")


def feature_extract_and_classification(patch, classifier, resize=(64, 64)):
    """对截取到的区域用SVM分类器分类
    Args:
        patch(np.ndarray):  矩形框截取的图像patch
        classifier(sklearn.svm._classes.SVC): 分类器
        resize(tuple):     resize大小
    Returns:
        cls_prob_: 属于各个类别的可能性
    """
    patch = cv2.resize(patch, resize)
    hogFeature = hog_feature(patch)
    # print("Feature Shape:", hogFeature.shape)
    hogFeature = np.reshape(hogFeature, (1, -1))

    cls_prob_ = classifier.predict_proba(hogFeature)
    return cls_prob_

def detect_single_image(image, classifier):
    """在单张图片上进行检测和分类
    Args:
        image(np.ndarray): 待检测的图片
        classifier(sklearn.svm._classes.SVC): SVM分类器
    """
    img_bbx = image.copy()
    rows, cols, _ = image.shape
    img_bin = convert_to_binary_img(image)
    rectangles = detect_contour(img_bin)

    for rect in rectangles:
        xCenter = int(rect[0] + rect[2]/2)
        yCenter = int(rect[1] + rect[3]/2)

        recSize = max(rect[2], rect[3])
        x1 = max(0, int(xCenter - recSize/2))
        x2 = max(cols, int(xCenter + recSize/2))
        y1 = max(0, int(yCenter - recSize/2))
        y2 = max(rows, int(yCenter + recSize/2))

        patch = image[y1:y2, x1:x2]
        class_prob = feature_extract_and_classification(patch, classifier)
        class_label = np.argmax(class_prob)
        class_name = cla_labels[int(class_label)]

        rect_color = None
        if class_name == "WBC":
            rect_color = (255, 0, 0)
        elif class_name == "RBC":
            rect_color = (0, 0, 255)
        elif class_name == "Pla":
            rect_color = (255, 0, 0)
        cv2.rectangle(img_bbx, (rect[0], rect[1]), (rect[0]+rect[2], rect[1]+rect[3]), rect_color, 2)
        cv2.putText(img_bbx, class_name, (rect[0], rect[1]), 1, 1.5, rect_color, 2)
    cv2.imshow("Bounded", img_bbx)
    cv2.waitKey(0)


def detect_BCCD_dir(classifier):
    [imgNames, _] = filterFiles(config.BCCD_JPEGImages, 'jpg')
    for img_ in imgNames:
        img_det = cv2.imread(config.BCCD_JPEGImages + img_)
        detect_single_image(image=img_det, classifier=classifier)
        # ESC 退出
        if cv2.waitKey() == 27:
            break


if __name__ == "__main__":
    """主程序"""
    if not os.path.exists(config.svmModel):
        [trainImgNames, trainLabels, trainHogs] = load_hog_and_label(config.train_hogFeaturePath, config.train_set_image)
        [testImgNames, testLabels, testHogs] = load_hog_and_label(config.test_hogFeaturePath, config.test_set_image)
        cls = SVMCls(trainHogs, trainLabels, testHogs, testLabels)
        cls.train()
        cls.evaluate_model()
        print(cls.test())
        print(testLabels.shape)

    img = cv2.imread(config.train_set_image+"Pla1.jpg")
    cls = joblib.load(config.svmModel)
    print(type(cls))
    cls_prob = feature_extract_and_classification(img, cls)
    print(type(cls_prob), cls_prob.shape, cls_prob)

    detect_BCCD_dir(cls)
