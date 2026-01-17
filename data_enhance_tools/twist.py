import os, cv2, math
import shutil

import numpy as np
from data_enhance_tools.perlin import Distorter
from data_enhance_tools.warpper import My_warpPerspective
from data_enhance_tools.mixup import My_mixup


class Twist:

    def __init__(self, to_gray, noise_config_path=r'.\temp',bj_img_path=r'E:\datasets\coco_datasets\coco_data'):

        name_cls_dict = {
            'title': '0',
            'author': '1',
            'TM': '2',
            'TempoNum': '3',
            'MN': '4',
            'TS': '5',
            'HX': '6',
            'pagemargin': '7',
            'Ins': '8',
            'Other': '9',
            'GC': '10',
            "Alter1": "11",
            "Alter2": "12"
        }

        self.cls_name_dict = {val: key for key, val in name_cls_dict.items()}

        self.color_type = {
            '0': (0, 0, 0),  # 黑色 title
            '1': (192, 192, 192),  # 灰色 author
            '2': (128, 128, 128),  # 深灰色 TM
            '3': (0, 0, 255),  # 红色 TempoNum
            '4': (0, 0, 128),  # 深红色 MN
            '5': (0, 255, 0),  # 绿色 TS
            '6': (0, 128, 0),  # 深绿色 HX
            '7': (255, 0, 0),  # 蓝色  pagemargin
            '8': (128, 0, 0),  # 深蓝色 Ins
            '9': (255, 0, 255),  # 紫红色 Other
            '10': (128, 0, 128),  # 深紫红 GC
            # '11':(0,255,255), # 黄色 Alter1
            '11': (64, 64, 64),  # 不知道什么颜色 Alter1
            '12': (0, 128, 128)  # 棕色 Alter2
        }

        self.arg_dict = {
            'scale': 0.1,
            'scale_sigma': 0.1,
            'intensity': 0.1,
            'intensity_sigma': 0.1,  # 扭曲系数
            'noise_weights_sigma': 0.1
        }

        self.noise_list = []
        for i in os.listdir(noise_config_path):
            self.noise_list.append(os.path.join(noise_config_path, i))

        self.distorter = Distorter(noise_path=self.noise_list)  # perlin噪声增强
        self.warpper = My_warpPerspective()  # 透射变换增强
        self.mixup = My_mixup(to_gray,bj_img_path)

    def tuple_sub(self, t1, t2):
        res = 0
        for i in zip(t1, t2):
            res += abs(i[0] - i[1])
        return res

    def judge_line_is_closed(self, line1, line2):
        '''
        :param line1:   (x1,x1)
        :param line2:   (x2,x2)
        :return:   True : 2直线距离很近    False: 2直线距离很远   （2直线均为x or y轴的投影）
        '''
        full_point = line1 + line2
        full_point = sorted(full_point)
        if (full_point[0] in line1 and full_point[1] in line2) or (
                full_point[0] in line2 and full_point[1] in line1):  # 表示2线段 有交点, 距离为负：表示很近
            return True
        elif full_point[2] - full_point[1] < 5:  # 2线段 无交点， 但是他们的距离 小于 5个像素 ： 表示很近
            return True

        return False

    def judge_box_is_closed(self, boxes):
        """
        :param boxes: [ [[x0,y0],[x1,y1],[x2,y2],[x3,y3]],[],[],.....]
        :return:   True  该文件中有 box之间的距离 太近（不适合后续的柏林噪声的数据增强）
                   False 该文件可以进行正常人的柏林噪声增强
        """
        for index1, i in enumerate(boxes):
            tmp0 = sorted(i, key=lambda x: x[0])
            x_range_1 = (tmp0[0][0], tmp0[-1][0])

            tmp1 = sorted(i, key=lambda x: x[1])
            y_range_1 = (tmp1[0][1], tmp1[-1][1])

            for index2, j in enumerate(boxes):
                if index1 != index2:
                    tmp2 = sorted(j, key=lambda x: x[0])
                    x_range_2 = (tmp2[0][0], tmp2[-1][0])

                    tmp3 = sorted(j, key=lambda x: x[0])
                    y_range_2 = (tmp3[0][1], tmp3[-1][1])

                    if self.judge_line_is_closed(x_range_1, x_range_2) and self.judge_line_is_closed(y_range_1,
                                                                                                     y_range_2):
                        return True

        return False

    def gen_label_img(self, train_img, txt_label_content):
        '''
        :param train_img:  numpy 格式的图像
        :param txt_label:  ['x0,y0,x1,y1,x2,y2,x3,y3,label\n',‘’，‘’]
        :return:
        '''

        gen_white_label_img = np.ones_like(train_img).astype(np.uint8) * 255
        boxes = []
        for content in txt_label_content:
            coord_con = content.split(',')
            x0, y0 = int(float(coord_con[0])), int(float(coord_con[1]))
            x1, y1 = int(float(coord_con[2])), int(float(coord_con[3]))
            x2, y2 = int(float(coord_con[4])), int(float(coord_con[5]))
            x3, y3 = int(float(coord_con[6])), int(float(coord_con[7]))
            pst = np.array([
                [x0, y0],
                [x1, y1],
                [x2, y2],
                [x3, y3],
            ])
            boxes.append(list(pst))
            cv2.fillPoly(gen_white_label_img, [pst], self.color_type[coord_con[8].strip()])

        is_to_close = self.judge_box_is_closed(boxes)  # 判断是否有2个文本距离太近
        if is_to_close is not True:  # 文本之间不是太近
            return gen_white_label_img
        return None

    def transformer(self, train_img, label_img=None):
        '''
        :param train_img:  训练图片
        :param label_img:  标签图片
        :return:  经过 perlin噪声 扭曲的 训练图片和标签图片
        '''
        #########################为了防止原图变换之后图像边缘的信息丢失，在原图上加白边，同时标签数据也要跟着改变#################################
        top, bottom, left, right = 0, 0, 0, 0
        cv_img = cv2.copyMakeBorder(train_img, top, bottom, left, right, cv2.BORDER_CONSTANT,
                                    value=(255, 255, 255))  # 给图像加白边

        if label_img is not None:
            label_img = cv2.copyMakeBorder(label_img, top, bottom, left, right, cv2.BORDER_CONSTANT,
                                           value=(255, 255, 255))  # 给图像加白边

        ###########################################################################################################################
        scale = self.arg_dict['scale'] * math.exp(np.random.randn() * self.arg_dict['scale_sigma'])  # shrink　
        intensity = self.arg_dict['intensity'] * math.exp(np.random.randn() * self.arg_dict['intensity_sigma'])
        nx, ny = self.distorter.make_maps(cv_img.shape, scale, intensity, self.arg_dict['noise_weights_sigma'])
        dealed_raw_img = self.distorter.distort(source=cv_img, mapx=nx, mapy=ny)
        if label_img is not None:
            dealed_label_img = self.distorter.distort(source=label_img, mapx=nx, mapy=ny)
            return dealed_raw_img, dealed_label_img
        return dealed_raw_img, None

    def do(self, cv_img, labels_list=None):
        ################ 与背景merge ############# begin
        # if np.random.uniform() > 0.6:
        cv_img, labels_list = self.mixup.do(cv_img, labels_list)
        ######################################## end

        ################ 透射变换 ################# begin
        # if np.random.uniform() > 0.5:
        cv_img, labels_list = self.warpper.do(cv_img, labels_list)
        ######################################### end

        n_train_img, _ = self.transformer(train_img = cv_img)

        return n_train_img


if __name__ == '__main__':
    save_count = 0
    save_img_path = r'D:\my_datasets\hx\user_realdatasets_buchong_2022_4_24_enhance'
    if not os.path.exists(save_img_path):
        os.makedirs(save_img_path)


    # img_path = r'E:\datasets\chord\book\chord_cut\all_samples_no_padding'  # 图片的路径

    # img_path = r'D:\my_datasets\tmp\tempnum\text_renderer-master-music-tempnum-2022-4-19\output\default'
    img_path = r'D:\my_datasets\hx\user_realdatasets_buchong_2022_4_24'
    support_img_format = ['.png', '.jpg']

    EPOCH = 2
    for epoch in range(EPOCH):
        for j in os.listdir(img_path):
            file_name, extend_name = os.path.splitext(j)
            if extend_name in support_img_format:
                detail_img_path = os.path.join(img_path, j)
                detail_label_path = os.path.join(img_path, j.replace(extend_name, '.txt'))

                save_detail_img_path = os.path.join(save_img_path, 'enhance%s_'%epoch + j)
                save_detail_label_path = os.path.join(save_img_path, 'enhance%s_'%epoch + j.replace(extend_name, '.txt'))

                img = cv2.imread(detail_img_path)

                twist = Twist()
                res = twist.do(img, labels_list=None)
                cv2.imwrite(save_detail_img_path, res)
                shutil.copy(detail_label_path, save_detail_label_path)
                save_count += 1
                print(f'save {save_count} 个 img。。。。')


                # cv2.imshow(f'{j}', res)
                # cv2.waitKey()
