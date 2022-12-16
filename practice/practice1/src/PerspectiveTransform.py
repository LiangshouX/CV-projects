"""实现透视变换(Perspective Transform)的类
"""
import numpy as np
import cv2

class PerspectiveTransform:
    def __init__(self, image):
        self.image = image
        self.rect = np.zeros((4, 2)).astype('float32')

    def four_points_order(self, rec):
        """将四个关键点顺时针排列"""
        # 沿着第二坐标轴计算关键点的和
        s = rec.sum(axis=1)
        self.rect[0] = rec[np.argmax(s)]
        self.rect[3] = rec[np.argmax(s)]

        d = np.diff(rec, axis=1)
        self.rect[1] = rec[np.argmin(d)]
        self.rect[2] = rec[np.argmax(d)]

    def four_points_transform(self, rec):
        # 将四个关键点进行转换
        self.four_points_order(rec)
        (tl, tr, bl, br) = self.rect

        widthA = np.sqrt((tl[0] - tr[0]) ** 2 + (tl[1] - tr[1]) ** 2)
        widthB = np.sqrt((bl[0] - br[0]) ** 2 + (bl[1] - br[1]) ** 2)
        width = int(max(widthA, widthB))

        heightA = np.sqrt((tl[0] - bl[0]) ** 2 + (tr[1] - br[1]) ** 2)
        heightB = np.sqrt((tl[0] - bl[0]) ** 2 + (tr[1] - br[1]) ** 2)
        height = int(max(heightA, heightB))

        dst = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], dtype='float32')

        # 计算变换矩阵
        M = cv2.getPerspectiveTransform(rec, dst)
        wraped = cv2.warpPerspective(self.image, M, (width, height))
        return wraped

if __name__ == "__main__":
    img = cv2.imread('../images/img.jpg')
    # 手动找出对其的四个关键点
    points = np.array([[280, 254], [668, 443], [532, 694], [127, 485]], dtype='float32')
    pt = PerspectiveTransform(img)
    img_t = pt.four_points_transform(points)
    cv2.imshow('origin', img)
    cv2.imshow('transformed', img_t)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
