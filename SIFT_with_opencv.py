import cv2 as cv
import matplotlib.pyplot as plt
from time import time

t1 = time()
img1 = cv.imread('./data/leuvenA.jpg')
img2 = cv.imread('./data/leuvenB.jpg')


def detect_sift(img):
    sift = cv.SIFT_create()  # SIFT特征提取对象
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)  # 转为灰度图
    kp = sift.detect(gray, None)  # 关键点位置
    kp, des = sift.compute(gray, kp)  # des为特征向量
    print(des.shape)  # 特征向量为128维
    return kp, des


kp1, des1 = detect_sift(img1)
kp2, des2 = detect_sift(img2)

bf = cv.BFMatcher(crossCheck=True)  # 匹配对象
matches = bf.match(des1, des2)  # 进行两个特征矩阵的匹配
res = cv.drawMatches(img1, kp1, img2, kp2, matches, None)  # 绘制匹配结果
plt.imshow(res)
t2 = time()
print("Matching finished with %.5f seconds" % (t2 - t1))
plt.show()
