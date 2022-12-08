import logging
import cv2
import numpy as np

from numpy import all, array, dot, stack, trace, floor, round, float32
from ComputeKeypointsWithOrientations import computeKeypointsWithOrientations

"""全局变量"""
logger = logging.getLogger(__name__)
float_tolerance = 1e-7

"""处理尺度空间的极值点"""

def findScaleSpaceExtrema(gaussian_images, dog_images, num_intervals, sigma, image_border_width,
                          contrast_threshold=0.04):
    """找出图像金字塔中所有尺度空间极值点的像素位置
    """
    logger.debug('Finding scale-space extrema...')
    # print("Finding scale-space extrema...\n")
    threshold = floor(0.5 * contrast_threshold / num_intervals * 255)  # from OpenCV implementation
    keypoints = []

    for octave_index, dog_images_in_octave in enumerate(dog_images):
        for image_index, (first_image, second_image, third_image) in enumerate(
                zip(dog_images_in_octave, dog_images_in_octave[1:], dog_images_in_octave[2:])):
            # (i, j) 是3x3矩阵的中心位置
            for i in range(image_border_width, first_image.shape[0] - image_border_width):
                for j in range(image_border_width, first_image.shape[1] - image_border_width):
                    if isPixelAnExtremum(first_image[i - 1:i + 2, j - 1:j + 2], second_image[i - 1:i + 2, j - 1:j + 2],
                                         third_image[i - 1:i + 2, j - 1:j + 2], threshold):
                        localization_result = localizeExtremumViaQuadraticFit(i, j, image_index + 1, octave_index,
                                                                              num_intervals, dog_images_in_octave,
                                                                              sigma, contrast_threshold,
                                                                              image_border_width)
                        if localization_result is not None:
                            keypoint, localized_image_index = localization_result
                            keypoints_with_orientations = computeKeypointsWithOrientations(keypoint, octave_index,
                                                                                           gaussian_images[
                                                                                               octave_index][
                                                                                               localized_image_index])
                            for keypoint_with_orientation in keypoints_with_orientations:
                                keypoints.append(keypoint_with_orientation)
    return keypoints


def isPixelAnExtremum(first_subimage, second_subimage, third_subimage, threshold):
    """判断像素点是否为极值：如果3x3x3输入数组的中心元素严格大于或小于其所有邻近元素，则返回True
    """
    center_pixel_value = second_subimage[1, 1]
    if abs(center_pixel_value) > threshold:
        if center_pixel_value > 0:
            return all(center_pixel_value >= first_subimage) and \
                   all(center_pixel_value >= third_subimage) and \
                   all(center_pixel_value >= second_subimage[0, :]) and \
                   all(center_pixel_value >= second_subimage[2, :]) and \
                   center_pixel_value >= second_subimage[1, 0] and \
                   center_pixel_value >= second_subimage[1, 2]
        elif center_pixel_value < 0:
            return all(center_pixel_value <= first_subimage) and \
                   all(center_pixel_value <= third_subimage) and \
                   all(center_pixel_value <= second_subimage[0, :]) and \
                   all(center_pixel_value <= second_subimage[2, :]) and \
                   center_pixel_value <= second_subimage[1, 0] and \
                   center_pixel_value <= second_subimage[1, 2]
    return False


def localizeExtremumViaQuadraticFit(i, j, image_index, octave_index, num_intervals, dog_images_in_octave, sigma,
                                    contrast_threshold, image_border_width, eigenvalue_ratio=10,
                                    num_attempts_until_convergence=5):
    """通过对每个极值相邻点的二次拟合迭代优化尺度空间极值点的像素位置
    """
    logger.debug("Localizing scale-space extrema...")
    # print("Localizing scale-space extrema...\n")
    extremum_is_outside_image = False
    image_shape = dog_images_in_octave[0].shape
    for attempt_index in range(num_attempts_until_convergence):
        # 需将像素点的数据类型从uint8 转换为 float32 以便计算导数；并且需要将像素值调整至[0,1]
        first_image, second_image, third_image = dog_images_in_octave[image_index - 1:image_index + 2]
        pixel_cube = stack([first_image[i - 1:i + 2, j - 1:j + 2],
                            second_image[i - 1:i + 2, j - 1:j + 2],
                            third_image[i - 1:i + 2, j - 1:j + 2]]).astype('float32') / 255.
        gradient = computeGradientAtCenterPixel(pixel_cube)
        hessian = computeHessianAtCenterPixel(pixel_cube)
        extremum_update = -np.linalg.lstsq(hessian, gradient, rcond=None)[0]
        if abs(extremum_update[0]) < 0.5 and abs(extremum_update[1]) < 0.5 and abs(extremum_update[2]) < 0.5:
            break
        j += int(round(extremum_update[0]))
        i += int(round(extremum_update[1]))
        image_index += int(round(extremum_update[2]))
        # 确保新的 像素点的立方 完全位于图像的范围内
        if i < image_border_width or i >= image_shape[0] - image_border_width or j < image_border_width or j >= \
                image_shape[1] - image_border_width or image_index < 1 or image_index > num_intervals:
            extremum_is_outside_image = True
            break
    if extremum_is_outside_image:
        logger.debug('Updated extremum moved outside of image before reaching convergence. Skipping...')
        return None
    if attempt_index >= num_attempts_until_convergence - 1:
        logger.debug('Exceeded maximum number of attempts without reaching convergence for this extremum. Skipping...')
        return None
    functionValueAtUpdatedExtremum = pixel_cube[1, 1, 1] + 0.5 * dot(gradient, extremum_update)
    if abs(functionValueAtUpdatedExtremum) * num_intervals >= contrast_threshold:
        xy_hessian = hessian[:2, :2]
        xy_hessian_trace = trace(xy_hessian)
        xy_hessian_det = np.linalg.det(xy_hessian)
        if xy_hessian_det > 0 and eigenvalue_ratio * (xy_hessian_trace ** 2) < (
                (eigenvalue_ratio + 1) ** 2) * xy_hessian_det:
            # 对比检查通过——构造并返回OpenCV KeyPoint对象
            keypoint = cv2.KeyPoint()
            keypoint.pt = (
                (j + extremum_update[0]) * (2 ** octave_index), (i + extremum_update[1]) * (2 ** octave_index))
            keypoint.octave = octave_index + image_index * (2 ** 8) + int(round((extremum_update[2] + 0.5) * 255)) * (
                    2 ** 16)
            keypoint.size = sigma * (2 ** ((image_index + extremum_update[2]) / float32(num_intervals))) * (
                    2 ** (octave_index + 1))  # octave_index + 1 because the input image was doubled
            keypoint.response = abs(functionValueAtUpdatedExtremum)
            return keypoint, image_index
    return None


def computeGradientAtCenterPixel(pixel_array):
    """利用O(h^2)阶中心差分公式，近似3x3x3阵列中心像素[1,1,1]处的梯度，其中h为步长
    """
    # 当步长为h时，f'(x)的O(h^2)阶中心差分公式为 f'(x) = (f(x + h) - f(x - h)) / (2 * h)
    # 此时h = 1,有 f'(x) = (f(x + 1) - f(x - 1)) / 2
    # X对应第二个数组轴，y对应第一个数组轴，s (scale)对应第三个数组轴
    dx = 0.5 * (pixel_array[1, 1, 2] - pixel_array[1, 1, 0])
    dy = 0.5 * (pixel_array[1, 2, 1] - pixel_array[1, 0, 1])
    ds = 0.5 * (pixel_array[2, 1, 1] - pixel_array[0, 1, 1])
    return array([dx, dy, ds])


def computeHessianAtCenterPixel(pixel_array):
    """利用O(h^2)阶中心差分公式近似3x3x3阵列中心像素[1,1,1]处的Hessian，其中h为步长
    """
    # f''(x) = f(x + 1) - 2 * f(x) + f(x - 1)
    # (d^2) f(x, y) / (dx dy) = (f(x + 1, y + 1) - f(x + 1, y - 1) - f(x - 1, y + 1) + f(x - 1, y - 1)) / 4
    # X对应第二个数组轴，y对应第一个数组轴，s (scale)对应第三个数组轴
    center_pixel_value = pixel_array[1, 1, 1]
    dxx = pixel_array[1, 1, 2] - 2 * center_pixel_value + pixel_array[1, 1, 0]
    dyy = pixel_array[1, 2, 1] - 2 * center_pixel_value + pixel_array[1, 0, 1]
    dss = pixel_array[2, 1, 1] - 2 * center_pixel_value + pixel_array[0, 1, 1]
    dxy = 0.25 * (pixel_array[1, 2, 2] - pixel_array[1, 2, 0] - pixel_array[1, 0, 2] + pixel_array[1, 0, 0])
    dxs = 0.25 * (pixel_array[2, 1, 2] - pixel_array[2, 1, 0] - pixel_array[0, 1, 2] + pixel_array[0, 1, 0])
    dys = 0.25 * (pixel_array[2, 2, 1] - pixel_array[2, 0, 1] - pixel_array[0, 2, 1] + pixel_array[0, 0, 1])
    return array([[dxx, dxy, dxs],
                  [dxy, dyy, dys],
                  [dxs, dys, dss]])
