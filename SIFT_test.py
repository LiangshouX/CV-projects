from matplotlib import pyplot as plt
import cv2
import numpy as np
import logging
import SIFT as S
from time import time

t1 = time()
print("starting...\n")

logger = logging.getLogger(__name__)

MIN_MATCH_COUNT = 10

img1 = cv2.imread('data/leuvenA.jpg', 0)
img2 = cv2.imread('data/leuvenA.jpg', 0)
print(type(img1))

# 计算SIFT关键点和描述符
kp1, des1 = S.computeKeypointsAndDescriptors(img1)
kp2, des2 = S.computeKeypointsAndDescriptors(img2)

print("图像特征提取完毕，开始匹配……\n")

FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)
flann = cv2.FlannBasedMatcher(index_params, search_params)
matches = flann.knnMatch(des1, des2, k=2)

# 劳氏比值判别法(Lowe's ratio test)
good = []
for m, n in matches:
    if m.distance < 0.7 * n.distance:
        good.append(m)

if len(good) > MIN_MATCH_COUNT:
    # 估计模板和场景之间的单应性
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    M = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)[0]

    # 在场景图像中绘制检测到的模板
    h, w = img1.shape
    pts = np.float32([[0, 0],
                      [0, h - 1],
                      [w - 1, h - 1],
                      [w - 1, 0]]).reshape(-1, 1, 2)
    dst = cv2.perspectiveTransform(pts, M)

    img2 = cv2.polylines(img2, [np.int32(dst)], True, 255, 3, cv2.LINE_AA)

    h1, w1 = img1.shape
    h2, w2 = img2.shape
    nWidth = w1 + w2
    nHeight = max(h1, h2)
    hdif = int((h2 - h1) / 2)
    newimg = np.zeros((nHeight, nWidth, 3), np.uint8)

    for i in range(3):
        newimg[hdif:hdif + h1, :w1, i] = img1
        newimg[:h2, w1:w1 + w2, i] = img2

    # 绘制SIFT关键点匹配
    for m in good:
        pt1 = (int(kp1[m.queryIdx].pt[0]), int(kp1[m.queryIdx].pt[1] + hdif))
        pt2 = (int(kp2[m.trainIdx].pt[0] + w1), int(kp2[m.trainIdx].pt[1]))
        cv2.line(newimg, pt1, pt2, (255, 0, 0))

    t2 = time()
    print("Matching finished with %.5f seconds" % (t2 - t1))
    plt.imshow(newimg)
    plt.show()
else:
    t2 = time()
    print("Not enough matches are found - %d/%d" % (len(good), MIN_MATCH_COUNT))
    print("Matching finished with %.5f seconds" % (t2 - t1))


