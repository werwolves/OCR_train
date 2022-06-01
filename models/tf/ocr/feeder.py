# -*- coding: utf8 -*-
import numpy as np

from util.data_feeder import DataFeeder
from util.image_utils2 import ImageTools, ImageRndUtils


class TFCtcDataFeeder(DataFeeder):
    def __init__(self, model, **kwargs):
        super().__init__(model, **kwargs)
        self.charset = self.model.model_def.charset
        self.rnd_proc_img = self.args.get('rnd_proc_img', "True")
        self.max_text_len = self.args.get('max_text_len',50)
        self.input_len_div = self.args.get('input_len_div',8)

    def process_batch(self, batch, is_test):
        size = len(batch)
        h, w, c = self.model.model_def.data_shape()
        batch_x = []
        batch_y = np.zeros((size, self.max_text_len),dtype=np.float32)
        batch_input_len = np.zeros((size, 1), dtype=np.int64)
        batch_label_len = np.zeros((size, 1), dtype=np.int64)

        i = 0
        for x in batch:
            im = x['im']
            if len(batch) == 1:
                im = ImageTools.resize_by_height(im, h)
                w = im.shape[1]
            else:
                 _, im = ImageTools.resize_and_pad_img(im, h, w)

                #_, im = ImageTools.resize_and_pad_img(im, 20, 60)
                #im = ImageRndUtils.updown_pad(im, (0, 0, 36, 0))

            if self.rnd_proc_img=="True":
                # im = ImageRndUtils.rnd_pad(im, (40, 9,40, 8)) # 银行卡
                # im = ImageRndUtils.rnd_pad(im, (16, 5, 0, 5)) # 公司默认代码
                # im = ImageRndUtils.rnd_pad(im, (16, 8, 16, 8))  # 左， 上， 右，下
                # im = ImageRndUtils.rnd_pad(im, (4, 3, 4, 3))  # 左， 上， 右，下
                im = ImageRndUtils.rnd_pad(im, (3, 2, 3, 2))  # 左， 上， 右，下

                # (l,t,r,b)
            # im = ImageRndUtils.updown_pad(im, (2, 4, 3, 4))
            
            # import cv2
            # cv2.imshow('tmp',im)
            # cv2.waitKey(0)

            text = x['label']
            y = self.charset.encode(text, self.max_text_len)
            if c == 1 and len(im.shape) == 2:
                im = np.expand_dims(im, 2)
            im = im.reshape((1,h,w,c)).astype(np.float32) / 255.0 - 0.5

            batch_x.append(im)
            batch_y[i] = y
            batch_input_len[i] = w // self.input_len_div
            batch_label_len[i] = len(text)

            i +=1
        batch_x = np.concatenate(batch_x, axis=0)

        if is_test:
            return {"input": batch_x, "input_length_1": batch_input_len}, batch
        else:
            inputs = {'input': batch_x,                     #  批次图片数据 归一化后 的图像数据
                      'label': batch_y,                     #  批次标签数据 《文本转向量》
                      'input_length': batch_input_len,     #  输入批量图像数据的长度  w//分成的份数
                      'label_length': batch_label_len,     #  输入的批量标签数据的长度
                      }

            outputs = {'ctc': np.zeros([size])}
            return (inputs, outputs)

