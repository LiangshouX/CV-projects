"""
@brief 建立图像的尺度空间

__IN__ numpy.ndarray类型的cv2图像
__OUT__  图像高斯模糊后得到的baseImage
        划分了倍频程的高斯金字塔Gaussian_Image
        由高斯金字塔构建得到的高斯差分矩阵DoG
"""

import logging
import cv2
import numpy as np
from numpy import array, log, round

logger = logging.getLogger(__name__)

def generateBaseImage(image, sigma, assumed_blur):
    """对输入进行双向上采样（采样率2）和模糊操作构建基础图像
    """
    logger.debug("Generating base image...")
    # print("Generating base image...\n")
    image = cv2.resize(src=image, dsize=(0, 0), fx=2, fy=2, interpolation=cv2.INTER_LINEAR)
    sigma_diff = np.sqrt(max((sigma ** 2) - ((2 * assumed_blur) ** 2), 0.01))
    # 返回后的图像模糊是sigma，而不是assumed_blur
    return cv2.GaussianBlur(src=image, ksize=(0, 0), sigmaX=sigma_diff,
                            sigmaY=sigma_diff)


def computeNumberOfOctaves(image_shape):
    """计算图像金字塔中的倍频程的数量作为基本图像形状的函数(OpenCV默认)
    """
    return int(round(log(min(image_shape)) / log(2) - 1))


def generateGaussianKernels(sigma, num_intervals):
    """生成一个高斯核列表，将用于模糊输入图象
    """
    logger.debug("Generating scales...")
    # print("Generating scales...\n")
    # 需要计算每个倍频程包含多少图像数量
    num_images_per_octave = num_intervals + 3
    # 常量因子k，计算某幅图像高斯平滑的标准差
    k = 2 ** (1. / num_intervals)
    gaussian_kernels = np.zeros(num_images_per_octave)
    gaussian_kernels[0] = sigma

    for image_index in range(1, num_images_per_octave):
        sigma_previous = (k ** (image_index - 1)) * sigma
        sigma_total = k * sigma_previous
        gaussian_kernels[image_index] = np.sqrt(sigma_total ** 2 - sigma_previous ** 2)
    return gaussian_kernels


def generateGaussianImages(image, num_octaves, gaussian_kernels):
    """高斯图像(金字塔)的尺度空间金字塔
    """
    logger.debug('Generating Gaussian images...')
    # print("Generating Gaussian images...\n")
    gaussian_images = []

    for octave_index in range(num_octaves):
        gaussian_images_in_octave = []
        # 倍频程中的第一幅图像已经有了正确的图像模糊
        gaussian_images_in_octave.append(image)
        for gaussian_kernel in gaussian_kernels[1:]:
            image = cv2.GaussianBlur(image, (0, 0), sigmaX=gaussian_kernel, sigmaY=gaussian_kernel)
            gaussian_images_in_octave.append(image)
        gaussian_images.append(gaussian_images_in_octave)
        octave_base = gaussian_images_in_octave[-3]
        image = cv2.resize(octave_base, (int(octave_base.shape[1] / 2), int(octave_base.shape[0] / 2)),
                           interpolation=cv2.INTER_NEAREST)
    return array(gaussian_images, dtype=object)


def generateDoGImages(gaussian_images):
    """建立高斯差分图像金字塔DoG
    """
    logger.debug('Generating Difference-of-Gaussian images...')
    # print("Generating Difference-of-Gaussian images...\n")
    dog_images = []

    for gaussian_images_in_octave in gaussian_images:
        dog_images_in_octave = []
        for first_image, second_image in zip(gaussian_images_in_octave, gaussian_images_in_octave[1:]):
            dog_images_in_octave.append(cv2.subtract(second_image,
                                                     first_image))
            # 不能使用普通减法，因为images是无符号整数
        dog_images.append(dog_images_in_octave)
    return array(dog_images, dtype=object)
