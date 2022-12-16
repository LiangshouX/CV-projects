"""实现图像分割的功能类"""
import cv2
import numpy as np

class ImageSegmentation:
    def __init__(self, img, method="k-means"):
        """实现图像分割"""
        self.img = img
        self.method = method
        assert method in ["k-means", "mean-shift"], "method should be in ['k-means', 'mean-shift']"

    def segmentation(self):
        pass

    def k_means_seg(self, k=2):
        """使用K-Means算法实现图像分割"""
        # 将二维像素转换为一维
        data = self.img.reshape((-1, 3)).astype("float32")

        # 定义中心[type, max_iter, epsilon]
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.)

        # 设置标签
        flags = cv2.KMEANS_RANDOM_CENTERS

        # K-Means聚类，根据k的值聚为k类
        compactness, labels, centers = cv2.kmeans(data=data, K=k,
                                                  bestLabels=None, criteria=criteria, attempts=10, flags=flags,)

        # 图像转换回二维uint8类型
        centers = centers.astype("uint8")
        # TODO

    def mean_shift_seg(self):
        """使用mean shift算法进行图像分割"""

if __name__ == "__main__":
    pass
