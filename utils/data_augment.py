# coding=utf-8
import cv2
import random
import numpy as np


class RandomGammaCorrection(object):
    def __init__(self):
        self.gamma = random.uniform(0.8, 1.3)

    def __call__(self, img, bbox):
        # 具体做法先归一化到1，然后gamma作为指数值求出新的像素值再还原
        gamma_table = [np.power(x / 255.0, self.gamma) * 255.0 for x in range(256)]
        gamma_table = np.round(np.array(gamma_table)).astype(np.uint8)
        img = cv2.LUT(img, gamma_table)
        return img, bbox


# 水平翻转
class RandomHorizontalFilp(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img, bboxes, img_path):
        if random.random() < self.p:
            _, w_img, _ = img.shape
            # img = np.fliplr(img)
            img = img[:, ::-1, :]
            bboxes[:, [0, 2]] = w_img - bboxes[:, [2, 0]]
        return img, bboxes


class GammaCorrection(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img, bboxes):
        if random.random() < self.p:
            return
        return img, bboxes


# 随机裁剪
class RandomCrop(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img, bboxes):
        if random.random() < self.p:
            h_img, w_img, _ = img.shape

            max_bbox = np.concatenate(
                [
                    np.min(bboxes[:, 0:2], axis=0),
                    np.max(bboxes[:, 2:4], axis=0),
                ],
                axis=-1,
            )
            max_l_trans = max_bbox[0]  # bbox最小的x坐标
            max_u_trans = max_bbox[1]  # bbox最小的y坐标
            max_r_trans = w_img - max_bbox[2]  # 图片宽度 - bbox最大的x坐标
            max_d_trans = h_img - max_bbox[3]  # 图片高度 - bbox最大的y坐标

            # 随机选取裁剪后的图片坐标
            crop_xmin = max(
                0, int(max_bbox[0] - random.uniform(0, max_l_trans))
            )
            crop_ymin = max(
                0, int(max_bbox[1] - random.uniform(0, max_u_trans))
            )
            crop_xmax = max(
                w_img, int(max_bbox[2] + random.uniform(0, max_r_trans))
            )
            crop_ymax = max(
                h_img, int(max_bbox[3] + random.uniform(0, max_d_trans))
            )

            img = img[crop_ymin:crop_ymax, crop_xmin:crop_xmax]  # 执行随机裁剪

            # 对bbox锚框进行坐标平移
            bboxes[:, [0, 2]] = bboxes[:, [0, 2]] - crop_xmin
            bboxes[:, [1, 3]] = bboxes[:, [1, 3]] - crop_ymin
        return img, bboxes


# 仿射变换
class RandomAffine(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img, bboxes):
        if random.random() < self.p:
            h_img, w_img, _ = img.shape
            # 得到可以包含所有bbox的最大bbox
            max_bbox = np.concatenate(
                [
                    np.min(bboxes[:, 0:2], axis=0),
                    np.max(bboxes[:, 2:4], axis=0),
                ],
                axis=-1,
            )
            max_l_trans = max_bbox[0]
            max_u_trans = max_bbox[1]
            max_r_trans = w_img - max_bbox[2]
            max_d_trans = h_img - max_bbox[3]

            # 平移变化 从 -(bbox最小x坐标 - 1) 到 (bbox最大坐标 - 1)
            tx = random.uniform(-(max_l_trans - 1), (max_r_trans - 1))
            ty = random.uniform(-(max_u_trans - 1), (max_d_trans - 1))

            M = np.array([[1, 0, tx], [0, 1, ty]])
            img = cv2.warpAffine(img, M, (w_img, h_img))  # 计算图片仿射变化(这里只进行了平移变换，没有旋转)

            bboxes[:, [0, 2]] = bboxes[:, [0, 2]] + tx
            bboxes[:, [1, 3]] = bboxes[:, [1, 3]] + ty
        return img, bboxes


# 缩放、通道置换以及像素值归一化
class Resize(object):
    """
    Resize the image to target size and transforms it into a color channel(BGR->RGB),
    as well as pixel value normalization([0,1])
    """

    def __init__(self, target_shape, correct_box=True):
        self.h_target, self.w_target = target_shape
        self.correct_box = correct_box

    def __call__(self, img, bboxes):
        h_org, w_org, _ = img.shape

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)  # 通道置换

        # 图片按最小比例缩放而非直接转化为指定尺寸
        resize_ratio = min(
            1.0 * self.w_target / w_org, 1.0 * self.h_target / h_org
        )
        resize_w = int(resize_ratio * w_org)
        resize_h = int(resize_ratio * h_org)
        image_resized = cv2.resize(img, (resize_w, resize_h))

        image_paded = np.full((self.h_target, self.w_target, 3), 128.0)
        dw = int((self.w_target - resize_w) / 2)
        dh = int((self.h_target - resize_h) / 2)
        image_paded[dh: resize_h + dh, dw: resize_w + dw, :] = image_resized
        image = image_paded / 255.0  # normalize to [0, 1]

        if self.correct_box:
            bboxes[:, [0, 2]] = bboxes[:, [0, 2]] * resize_ratio + dw
            bboxes[:, [1, 3]] = bboxes[:, [1, 3]] * resize_ratio + dh
            return image, bboxes
        return image


# 随机图像滤波
class RandomBlur(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img_org, bboxes_org):
        if random.random() > self.p:
            img_org = cv2.GaussianBlur(img_org, (3, 3), 0)
            img_org = img_org.astype(np.float32)
        return img_org, bboxes_org


def rand_uniform_strong(min, max):
    if min > max:
        swap = min
        min = max
        max = swap
    return random.random() * (max - min) + min


def rand_scale(s):
    scale = rand_uniform_strong(1, s)
    if random.randint(0, 1) % 2:
        return scale
    return 1. / scale


class RandomHSV(object):
    def __init__(self, hue=0.1, saturation=1.5, exposure=1.5, p=0.5):
        self.hue = hue
        self.saturation = saturation
        self.exposure = exposure
        self.p = p

    def __call__(self, img_org, bboxes_org):
        if random.random() < self.p:
            return img_org, bboxes_org
        dhue = rand_uniform_strong(-self.hue, self.hue)  # 色调
        dsat = rand_scale(self.saturation)  # 饱和度
        dexp = rand_scale(self.exposure)  # 曝光度
        if dsat != 1 or dexp != 1 or dhue != 0:
            if img_org.shape[2] >= 3:
                hsv_src = cv2.cvtColor(img_org.astype(np.float32), cv2.COLOR_BGR2HSV)  # RGB to HSV
                hsv = cv2.split(hsv_src)
                hsv[1] *= dsat
                hsv[2] *= dexp
                hsv[0] += 179 * dhue
                hsv_src = cv2.merge(hsv)
                img_org = np.clip(cv2.cvtColor(hsv_src, cv2.COLOR_HSV2RGB), 0, 255)  # HSV to RGB (the same as previous)
            else:
                img_org *= dexp
        img = img_org.astype(np.float32)
        return img, bboxes_org


# Mixup图片融合Trick
class Mixup(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img_org, bboxes_org, img_mix, bboxes_mix):
        if random.random() > self.p:
            lam = np.random.beta(1.5, 1.5)  # 生成B分布的随机数
            img = lam * img_org + (1 - lam) * img_mix
            bboxes_org = np.concatenate(
                [bboxes_org, np.full((len(bboxes_org), 1), lam)], axis=1
            )
            bboxes_mix = np.concatenate(
                [bboxes_mix, np.full((len(bboxes_mix), 1), 1 - lam)], axis=1
            )
            bboxes = np.concatenate([bboxes_org, bboxes_mix])

        else:
            img = img_org
            bboxes = np.concatenate(
                [bboxes_org, np.full((len(bboxes_org), 1), 1.0)], axis=1
            )

        return img, bboxes


# 正则化策略用来防止过拟合
class LabelSmooth(object):
    def __init__(self, delta=0.01):
        self.delta = delta

    def __call__(self, onehot, num_classes):
        return onehot * (1 - self.delta) + self.delta * 1.0 / num_classes


if __name__ == "__main__":
    img_org = cv2.imread("D:/Python_Project/raw_image.jpg")
    cv2.imshow("img_org", img_org)
    img, _ = RandomHSV()(img_org, 1)
    cv2.imshow("img_hsv", img)
    cv2.waitKey(0)
