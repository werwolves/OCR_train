# -*- coding: utf8 -*-
import io
import random

import cv2
import numpy as np
from PIL import Image, ImageEnhance
from scipy import misc


class ImageTools:
    @staticmethod
    def is_color_image(im):
        return len(im.shape) > 2 and im.shape[2] > 1

    @staticmethod
    def is_gray_image(im):
        return len(im.shape) == 2 or (len(im.shape) > 2 and im.shape[2] == 1)

    @staticmethod
    def convert_to_gray(im):
        if ImageTools.is_color_image(im):
            return cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
        if im.dtype != np.uint8:
            return im.astype(np.uint8)
        return im

    @staticmethod
    def convert_gray_to_rgb(im):
        if ImageTools.is_gray_image(im):
            return cv2.cvtColor(im, cv2.COLOR_GRAY2RGB)
        if im.dtype != np.uint8:
            return im.astype(np.uint8)
        return im

    @staticmethod
    def resize_by_height(im, h):
        '''
        根据高度拉伸图像
        :param im:
        :param h:
        :return:
        '''
        im_h, im_w = im.shape[:2]
        if im_h != h:
            new_w = int(1.0 * h / im_h * im_w)
            return ImageTools.resize_im(im, (h, new_w), interp='bicubic')
            # return cv2.resize(im, (new_w, h))
        return im

    @staticmethod
    def resize_rnd_method(im, size):
        # methods = ['nearest', 'lanczos', 'bilinear', 'bicubic' , 'cubic']
        return np.array(Image.fromarray(im).resize((size[1], size[0]), resample=random.choice(range(4))))

    @staticmethod
    def resize_im(im, size, interp='bicubic'):
        methods = ['nearest', 'lanczos', 'bilinear', 'bicubic', 'cubic']
        return np.array(Image.fromarray(im).resize((size[1], size[0]), resample=methods.index(interp)))

    @staticmethod
    def resize(im, h, w):
        '''
        强制拉伸图像
        :param im:
        :param h:
        :param w:
        :return:
        '''
        im_h, im_w = im.shape[:2]
        if im_h != h or im_w != w:
            return np.array(Image.fromarray(im).resize((w, h), resample= 3))
        return im

    @staticmethod
    def resize_and_pad(im, h, w):
        '''
        根据height/width等比缩放图像，剩余部分填充空白
        :param im:
        :param w:
        :return:
        '''
        im_h, im_w = im.shape[:2]
        if im_h == h and im_w == w:
            return (im, 1.0)

        sy = 1.0 * h / im_h
        sx = 1.0 * w / im_w
        s = min(sx, sy)

        new_h = int(s * im_h)
        new_w = int(s * im_w)

        # func = {'nearest': 0, 'lanczos': 1, 'bilinear': 2, 'bicubic': 3, 'cubic': 3}
        # imnew = im.resize((new_w, new_h), resample=func['bicubic'])

        # im = misc.imresize(im, (new_h, new_w), interp='bicubic')
        im = np.array(Image.fromarray(im).resize((new_w, new_h), resample=3))

        pad_r = w - new_w
        pad_b = h - new_h
        if pad_b > 0 or pad_r > 0:
            if len(im.shape) == 2:
                im = np.pad(im, ((0, pad_b), (0, pad_r)), mode='constant', constant_values=(255, 255))
            else:
                im = np.pad(im, ((0, pad_b), (0, pad_r), (0, 0)), mode='constant', constant_values=(255, 255))

        return (im, s)

    @staticmethod
    def pad_right(im,w):
        im_h, im_w = im.shape[:2]
        pad_r = w - im_w
        if len(im.shape) == 2:
            im = np.pad(im, ((0, 0), (0, pad_r)), mode='constant', constant_values=(255, 255))
        else:
            im = np.pad(im, ((0, 0), (0, pad_r), (0, 0)), mode='constant', constant_values=(255, 255))
        return im

    @staticmethod
    def save_im_to_jpg_bytes(mat):
        with Image.fromarray(mat) as im:
            with io.BytesIO() as stream:
                im.save(stream, format='JPEG')
                return stream.getvalue()

    @staticmethod
    def save_im_to_file(mat, file):
        # misc.imsave(file,mat)
        with Image.fromarray(mat) as im:
            im.save(file, format='JPEG')

    @staticmethod
    def open_from_bytes(img_bytes, to_gray=True):
        with io.BytesIO(img_bytes) as fs:
            with Image.open(fs) as im:
                if to_gray:
                    im = im.convert('L')
                else:
                    im = im.convert('RGB')
                return np.array(im)


    @staticmethod
    def resize_and_pad_img(im, h, fill_width, pad_left_per=0):
        '''
        根据高度缩放图像，宽度右边补充白色背景
        :param self:
        :param im: 数像：nparray
        :param h:  新高度
        :param fill_width: 填充后的宽度
        :return: 返回填充背景前的宽度及影像对象
        '''

        im_h, im_w = im.shape[:2]

        if im_h != h or im_w > fill_width:
            im, _s = ImageTools.resize_and_pad(im, h, fill_width)
            im_h, im_w = im.shape[:2]

        if im_w > fill_width:
            # 超过最大宽度
            return (im_w, im)
        elif im_w == fill_width:
            return (im_w, im)

        # 白色背景填充
        pad_w = fill_width - im_w

        if pad_left_per < 0:
            pad_left = random.randint(0, pad_w)
            pad_right = pad_w - pad_left
        else:
            pad_left = int(pad_w * pad_left_per)
            pad_right = pad_w - pad_left

        im = np.pad(im, ((0, 0), (pad_left, pad_right)), mode='constant', constant_values=(255, 255))
        return (im_w, im)

    @staticmethod
    def resize_or_pad_img(im, h, w, pad_per=(0.5, 0.5)):
        '''
        缩放或填充图像（大图按高度缩小，小图填充）
        :param h:  目标高度
        :param w:  目标宽度
        :param pad_per:  填充比像（高度,宽度)
        :return:
        '''

        im_h, im_w = im.shape[:2]

        if im_h > h:
            im = ImageTools.resize_by_height(im, h)
            im_h, im_w = im.shape[:2]

        # 白色背景填充
        pad_w = w - im_w if w > im_w else 0
        pad_h = h - im_h if h > im_h else 0

        if pad_h == 0 and pad_w == 0:
            return (im_w, im)

        if pad_per[0] < 0:
            pad_top = random.randint(0, pad_h)
            pad_bottom = pad_h - pad_top
        else:
            pad_top = int(pad_h * pad_per[0])
            pad_bottom = pad_h - pad_top

        if pad_per[1] < 0:
            pad_left = random.randint(0, pad_w)
            pad_right = pad_w - pad_left
        else:
            pad_left = int(pad_w * pad_per[1])
            pad_right = pad_w - pad_left

        im = ImageTools.pad_img(im, (pad_top, pad_right, pad_bottom, pad_left))
        return (im_w, im)

    @staticmethod
    def open_from_file(image_path, to_gray):
        im = np.array(Image.open(image_path))
        if to_gray:
            im = ImageTools.convert_to_gray(im)
        elif im.shape[-1] == 4:
            im = np.array(Image.fromarray(im).convert('RGB'))

        return im

    @staticmethod
    def pad_img(im, padding, fill_value=(255, 255)):
        t, r, b, l = padding

        if len(im.shape) == 3:
            im = np.pad(im, ((t, b), (l, r), (0, 0)), mode='constant', constant_values=fill_value)
        else:
            im = np.pad(im, ((t, b), (l, r)), mode='constant',
                        constant_values=fill_value)
        return im

    @staticmethod
    def rotate_img(im, angle):
        im = Image.fromarray(im)
        im = im.rotate(angle,resample=3)
        return np.array(im)

class YoloBoxUtils:
    @staticmethod
    def box_xywh_to_xyxy(box):
        shape = box.shape
        assert shape[-1] == 4, "Box shape[-1] should be 4."

        box = box.reshape((-1, 4))
        box[:, 0], box[:, 2] = box[:, 0] - box[:, 2] / 2, box[:, 0] + box[:, 2] / 2
        box[:, 1], box[:, 3] = box[:, 1] - box[:, 3] / 2, box[:, 1] + box[:, 3] / 2
        box = box.reshape(shape)
        return box

    @staticmethod
    def box_iou_xywh(box1, box2):
        assert box1.shape[-1] == 4, "Box1 shape[-1] should be 4."
        assert box2.shape[-1] == 4, "Box2 shape[-1] should be 4."

        b1_x1, b1_x2 = box1[:, 0] - box1[:, 2] / 2, box1[:, 0] + box1[:, 2] / 2
        b1_y1, b1_y2 = box1[:, 1] - box1[:, 3] / 2, box1[:, 1] + box1[:, 3] / 2
        b2_x1, b2_x2 = box2[:, 0] - box2[:, 2] / 2, box2[:, 0] + box2[:, 2] / 2
        b2_y1, b2_y2 = box2[:, 1] - box2[:, 3] / 2, box2[:, 1] + box2[:, 3] / 2

        inter_x1 = np.maximum(b1_x1, b2_x1)
        inter_x2 = np.minimum(b1_x2, b2_x2)
        inter_y1 = np.maximum(b1_y1, b2_y1)
        inter_y2 = np.minimum(b1_y2, b2_y2)
        inter_w = inter_x2 - inter_x1 + 1
        inter_h = inter_y2 - inter_y1 + 1
        inter_w[inter_w < 0] = 0
        inter_h[inter_h < 0] = 0

        inter_area = inter_w * inter_h
        b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
        b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)

        return inter_area / (b1_area + b2_area - inter_area)

    @staticmethod
    def box_iou_xyxy(box1, box2):
        assert box1.shape[-1] == 4, "Box1 shape[-1] should be 4."
        assert box2.shape[-1] == 4, "Box2 shape[-1] should be 4."

        b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]

        inter_x1 = np.maximum(b1_x1, b2_x1)
        inter_x2 = np.minimum(b1_x2, b2_x2)
        inter_y1 = np.maximum(b1_y1, b2_y1)
        inter_y2 = np.minimum(b1_y2, b2_y2)
        inter_w = inter_x2 - inter_x1
        inter_h = inter_y2 - inter_y1
        inter_w[inter_w < 0] = 0
        inter_h[inter_h < 0] = 0

        inter_area = inter_w * inter_h
        b1_area = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
        b2_area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)

        return inter_area / (b1_area + b2_area - inter_area)

    @staticmethod
    def box_crop(boxes, labels, scores, crop, img_shape):
        x, y, w, h = map(float, crop)
        im_w, im_h = map(float, img_shape)

        boxes = boxes.copy()
        boxes[:, 0], boxes[:, 2] = (boxes[:, 0] - boxes[:, 2] / 2) * im_w, (boxes[:, 0] + boxes[:, 2] / 2) * im_w
        boxes[:, 1], boxes[:, 3] = (boxes[:, 1] - boxes[:, 3] / 2) * im_h, (boxes[:, 1] + boxes[:, 3] / 2) * im_h

        crop_box = np.array([x, y, x + w, y + h])
        centers = (boxes[:, :2] + boxes[:, 2:]) / 2.0
        mask = np.logical_and(crop_box[:2] <= centers, centers <= crop_box[2:]).all(axis=1)

        boxes[:, :2] = np.maximum(boxes[:, :2], crop_box[:2])
        boxes[:, 2:] = np.minimum(boxes[:, 2:], crop_box[2:])
        boxes[:, :2] -= crop_box[:2]
        boxes[:, 2:] -= crop_box[:2]

        mask = np.logical_and(mask, (boxes[:, :2] < boxes[:, 2:]).all(axis=1))
        boxes = boxes * np.expand_dims(mask.astype('float32'), axis=1)
        labels = labels * mask.astype('float32')
        scores = scores * mask.astype('float32')
        boxes[:, 0], boxes[:, 2] = (boxes[:, 0] + boxes[:, 2]) / 2 / w, (boxes[:, 2] - boxes[:, 0]) / w
        boxes[:, 1], boxes[:, 3] = (boxes[:, 1] + boxes[:, 3]) / 2 / h, (boxes[:, 3] - boxes[:, 1]) / h

        return boxes, labels, scores, mask.sum()


class ImageRndUtils:
    @staticmethod
    def random_distort(img):
        def random_brightness(img, lower=0.5, upper=1.5):
            e = np.random.uniform(lower, upper)
            return ImageEnhance.Brightness(img).enhance(e)

        def random_contrast(img, lower=0.5, upper=1.5):
            e = np.random.uniform(lower, upper)
            return ImageEnhance.Contrast(img).enhance(e)

        def random_color(img, lower=0.5, upper=1.5):
            e = np.random.uniform(lower, upper)
            return ImageEnhance.Color(img).enhance(e)

        ops = [random_brightness, random_contrast, random_color]
        np.random.shuffle(ops)

        img = Image.fromarray(img)
        img = ops[0](img)
        img = ops[1](img)
        img = ops[2](img)
        img = np.asarray(img)

        return img

    @staticmethod
    def rnd_pad(img, margin, fill_color=(255, 255)):
        im = Image.fromarray(img)
        w, h = im.size
        l = random.randint(0, margin[0])
        t = random.randint(0, margin[1])
        r = random.randint(0, margin[2])
        b = random.randint(0, margin[3])

        if len(img.shape) == 2:
            im = np.pad(im, ((t, b), (l, r)), mode='constant', constant_values=fill_color)
        else:
            im = np.pad(im, ((t, b), (l, r), (0, 0)), mode='constant', constant_values=fill_color)

        im = Image.fromarray(im)
        im = im.resize((w, h), resample=random.randint(0,5))
        #im.save('./data/tmp.jpg', format='JPEG')
        return np.array(im)

    @staticmethod
    def random_crop(img,
                    boxes,
                    labels,
                    scores,
                    scales=[0.3, 1.0],
                    max_ratio=2.0,
                    constraints=None,
                    max_trial=50):
        if len(boxes) == 0:
            return img, boxes

        if not constraints:
            constraints = [
                (0.1, 1.0),
                (0.3, 1.0),
                (0.5, 1.0),
                (0.7, 1.0),
                (0.9, 1.0),
                (0.0, 1.0)]

        img = Image.fromarray(img)
        w, h = img.size
        crops = [(0, 0, w, h)]
        for min_iou, max_iou in constraints:
            for _ in range(max_trial):
                scale = random.uniform(scales[0], scales[1])
                aspect_ratio = random.uniform(max(1 / max_ratio, scale * scale), \
                                              min(max_ratio, 1 / scale / scale))
                crop_h = int(h * scale / np.sqrt(aspect_ratio))
                crop_w = int(w * scale * np.sqrt(aspect_ratio))
                crop_x = random.randrange(w - crop_w)
                crop_y = random.randrange(h - crop_h)
                crop_box = np.array([[
                    (crop_x + crop_w / 2.0) / w,
                    (crop_y + crop_h / 2.0) / h,
                    crop_w / float(w),
                    crop_h / float(h)
                ]])

                iou = YoloBoxUtils.box_iou_xywh(crop_box, boxes)
                if min_iou <= iou.min() and max_iou >= iou.max():
                    crops.append((crop_x, crop_y, crop_w, crop_h))
                    break

        while crops:
            crop = crops.pop(np.random.randint(0, len(crops)))
            crop_boxes, crop_labels, crop_scores, box_num = \
                YoloBoxUtils.box_crop(boxes, labels, scores, crop, (w, h))
            if box_num < 1:
                continue
            img = img.crop((crop[0], crop[1], crop[0] + crop[2],
                            crop[1] + crop[3])).resize(img.size, Image.LANCZOS)
            img = np.asarray(img)
            return img, crop_boxes, crop_labels, crop_scores
        img = np.asarray(img)
        return img, boxes, labels, scores

    @staticmethod
    def random_flip(img, gtboxes, thresh=0.5):
        if random.random() <= thresh:
            return img, gtboxes

        if len(img.shape) == 3:
            img = img[:, ::-1, :]
        else:
            img = img[:, ::-1]

        gtboxes[:, 0] = 1.0 - gtboxes[:, 0]
        return img, gtboxes

    @staticmethod
    def random_interp(img, size, interp=None):
        interp_method = [
            cv2.INTER_NEAREST,
            cv2.INTER_LINEAR,
            cv2.INTER_AREA,
            cv2.INTER_CUBIC,
            cv2.INTER_LANCZOS4,
        ]
        if not interp or interp not in interp_method:
            interp = interp_method[random.randint(0, len(interp_method) - 1)]
        h, w = img.shape[:2]
        im_scale_x = size / float(w)
        im_scale_y = size / float(h)
        img = cv2.resize(img, None, None, fx=im_scale_x, fy=im_scale_y,
                         interpolation=interp)
        return img

    @staticmethod
    def random_expand(img,
                      gtboxes,
                      max_ratio=2.,
                      fill=None,
                      keep_ratio=True,
                      thresh=0.5):
        if random.random() > thresh:
            return img, gtboxes

        if max_ratio < 1.0:
            return img, gtboxes

        if len(img.shape) == 2:
            img = np.expand_dims(img, axis=2)

        h, w, c = img.shape
        ratio_x = random.uniform(1, max_ratio)
        if keep_ratio:
            ratio_y = ratio_x
        else:
            ratio_y = random.uniform(1, max_ratio)
        oh = int(h * ratio_y)
        ow = int(w * ratio_x)
        off_x = random.randint(0, ow - w)
        off_y = random.randint(0, oh - h)

        out_img = np.zeros((oh, ow, c))
        if fill and len(fill) == c:
            for i in range(c):
                out_img[:, :, i] = fill[i] * 255.0

        out_img[off_y: off_y + h, off_x: off_x + w, :] = img
        gtboxes[:, 0] = ((gtboxes[:, 0] * w) + off_x) / float(ow)
        gtboxes[:, 1] = ((gtboxes[:, 1] * h) + off_y) / float(oh)
        gtboxes[:, 2] = gtboxes[:, 2] / ratio_x
        gtboxes[:, 3] = gtboxes[:, 3] / ratio_y

        out_img = np.asarray(out_img)
        if c == 1:
            out_img = np.reshape(out_img, out_img.shape[:2])

        return out_img, gtboxes
        # return out_img.astype('uint8'), gtboxes

    @staticmethod
    def shuffle_gtbox(gtbox, gtlabel, gtscore):
        gt = np.concatenate([gtbox, gtlabel[:, np.newaxis],
                             gtscore[:, np.newaxis]], axis=1)
        idx = np.arange(gt.shape[0])
        np.random.shuffle(idx)
        gt = gt[idx, :]
        return gt[:, :4], gt[:, 4], gt[:, 5]

    @staticmethod
    def image_mixup(img1,
                    gtboxes1,
                    gtlabels1,
                    gtscores1,
                    img2,
                    gtboxes2,
                    gtlabels2,
                    gtscores2):
        factor = np.random.beta(1.5, 1.5)
        factor = max(0.0, min(1.0, factor))
        if factor >= 1.0:
            return img1, gtboxes1, gtlabels1
        if factor <= 0.0:
            return img2, gtboxes2, gtlabels2
        gtscores1 = gtscores1 * factor
        gtscores2 = gtscores2 * (1.0 - factor)

        h = max(img1.shape[0], img2.shape[0])
        w = max(img1.shape[1], img2.shape[1])
        img = np.zeros((h, w, img1.shape[2]), 'float32')
        img[:img1.shape[0], :img1.shape[1], :] = img1.astype('float32') * factor
        img[:img2.shape[0], :img2.shape[1], :] += \
            img2.astype('float32') * (1.0 - factor)
        gtboxes = np.zeros_like(gtboxes1)
        gtlabels = np.zeros_like(gtlabels1)
        gtscores = np.zeros_like(gtscores1)

        gt_valid_mask1 = np.logical_and(gtboxes1[:, 2] > 0, gtboxes1[:, 3] > 0)
        gtboxes1 = gtboxes1[gt_valid_mask1]
        gtlabels1 = gtlabels1[gt_valid_mask1]
        gtscores1 = gtscores1[gt_valid_mask1]
        gtboxes1[:, 0] = gtboxes1[:, 0] * img1.shape[1] / w
        gtboxes1[:, 1] = gtboxes1[:, 1] * img1.shape[0] / h
        gtboxes1[:, 2] = gtboxes1[:, 2] * img1.shape[1] / w
        gtboxes1[:, 3] = gtboxes1[:, 3] * img1.shape[0] / h

        gt_valid_mask2 = np.logical_and(gtboxes2[:, 2] > 0, gtboxes2[:, 3] > 0)
        gtboxes2 = gtboxes2[gt_valid_mask2]
        gtlabels2 = gtlabels2[gt_valid_mask2]
        gtscores2 = gtscores2[gt_valid_mask2]
        gtboxes2[:, 0] = gtboxes2[:, 0] * img2.shape[1] / w
        gtboxes2[:, 1] = gtboxes2[:, 1] * img2.shape[0] / h
        gtboxes2[:, 2] = gtboxes2[:, 2] * img2.shape[1] / w
        gtboxes2[:, 3] = gtboxes2[:, 3] * img2.shape[0] / h

        gtboxes_all = np.concatenate((gtboxes1, gtboxes2), axis=0)
        gtlabels_all = np.concatenate((gtlabels1, gtlabels2), axis=0)
        gtscores_all = np.concatenate((gtscores1, gtscores2), axis=0)
        gt_num = min(len(gtboxes), len(gtboxes_all))
        gtboxes[:gt_num] = gtboxes_all[:gt_num]
        gtlabels[:gt_num] = gtlabels_all[:gt_num]
        gtscores[:gt_num] = gtscores_all[:gt_num]
        return img.astype('uint8'), gtboxes, gtlabels, gtscores

    @staticmethod
    def image_augment(img, gtboxes, gtlabels, gtscores, size, means=None):

        # h,w = img.shape[:2]
        # if h>size[0] and w > size[1]:
        #     img, gtboxes =  ImageRndUtils.random_expand(img, gtboxes, fill=means)
        #
        # img, gtboxes, gtlabels, gtscores = ImageRndUtils.random_crop(img, gtboxes, gtlabels, gtscores)
        h, w = img.shape[:2]
        if h != size[0] or w != size[1]:
            img, _ = ImageTools.resize_and_pad(img, size[0], size[1])
            # img = ImageTools.resize_rnd_method(img, size)

        img = ImageRndUtils.random_distort(img)

        # img, gtboxes = ImageRndUtils.random_flip(img, gtboxes)
        # gtboxes, gtlabels, gtscores = ImageRndUtils.shuffle_gtbox(gtboxes, gtlabels, gtscores)

        return img.astype('float32'), gtboxes.astype('float32'), gtlabels.astype('int32'), gtscores.astype('float32')
