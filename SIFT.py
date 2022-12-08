import logging
from GenerateScales import generateBaseImage, computeNumberOfOctaves, generateGaussianKernels, \
    generateGaussianImages, generateDoGImages
from LookForExtrema import findScaleSpaceExtrema
from ComputeKeypointsWithOrientations import removeDuplicateKeypoints
from KeypointsToImages import convertKeypointsToInputImageSize
from GenerateDescriptors import generateDescriptors

"""全局变量"""
logger = logging.getLogger(__name__)
float_tolerance = 1e-7

"""SIFT算法的主函数"""


def computeKeypointsAndDescriptors(image, sigma=1.6, num_intervals=3, assumed_blur=0.5, image_border_width=5):
    image = image.astype('float32')
    base_image = generateBaseImage(image, sigma, assumed_blur)  # 构建图像金字塔
    num_octaves = computeNumberOfOctaves(base_image.shape)  # 计算倍频程
    gaussian_kernels = generateGaussianKernels(sigma, num_intervals)  # 高斯核是一个可变尺度的高斯核
    gaussian_images = generateGaussianImages(base_image, num_octaves, gaussian_kernels)
    dog_images = generateDoGImages(gaussian_images)  # 计算高斯差分矩阵
    keypoints = findScaleSpaceExtrema(gaussian_images, dog_images, num_intervals, sigma, image_border_width)
    keypoints = removeDuplicateKeypoints(keypoints)
    keypoints = convertKeypointsToInputImageSize(keypoints)
    descriptors = generateDescriptors(keypoints, gaussian_images)
    return keypoints, descriptors
