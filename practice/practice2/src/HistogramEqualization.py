"""直方图均衡化(Histogram Equalization)功能类
        * 读入一幅灰度图像
        * 进行直方图均衡化
        * 显示结果
"""
import matplotlib.pyplot as plt
import numpy as np
import cv2

class HistogramEqualization:
    def __init__(self, img_gray, level=256):
        """HE类
        Args:
            img_gray(numpy.array): 输入的灰度图像
            level(int): 256
        """
        self.img_gray = img_gray
        self.level = level

    def HE(self, **args):
        """直方图均衡化"""
        # 计算直方图
        hists = self.cal_histogram()
        self.draw_histogram(hists=hists, fig_name="origin")

        # 均衡化
        m, n = self.img_gray.shape
        hists_cdf = self.cal_histogram_cdf(hists, m, n)
        self.draw_histogram(hists=hists_cdf, fig_name="cdf")

        # arr = np.zeros_like(self.img_gray)
        arr = hists_cdf[self.img_gray]

        return arr

    def cal_histogram(self):
        """计算灰度图的直方图
        Args:

        Returns:
            hists(list):
        """
        hists = np.zeros(self.level)
        for row in self.img_gray:
            for col in row:
                hists[col] += 1
        return hists

    def cal_histogram_cdf(self, hists, block_m, block_n):
        """计算累计分布函数CDF
        """
        hists_cumsum = np.cumsum(hists)
        const_a = (self.level - 1) / (block_m * block_n)
        hists_cdf = (const_a * hists_cumsum).astype("uint8")
        return hists_cdf

    def draw_histogram(self, hists, fig_name):
        """绘制直方图"""
        plt.figure()
        plt.bar(range(hists.size), hists)
        plt.show()
        plt.savefig("../image"+fig_name)


if __name__ == "__main__":
    # 读入图片
    img = cv2.imread("../images/car.png")

    # 将图片转换为灰度图
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    print(img_gray.shape)

    he = HistogramEqualization(img_gray)
    img1 = he.HE()
    cv2.imshow("Origin", img_gray)
    cv2.imshow("HE", img1)
    cv2.waitKey(5000)
