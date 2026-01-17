import os.path
import cv2
from glob import glob
import numpy as np
import random
from data_enhance_tools.textureSet import TextureSet

class My_mixup:

    def __init__(self, to_gray=False, bj_img_path=r'E:\datasets\coco_datasets\coco_data'):
        '''
        :param bj_img_path:  背景图的素材地址
        '''
        self.to_gray = to_gray
        self.textures = TextureSet(bj_img_path)
        self.iter = self.textures.makeIterator(frame_size = (512, 512), frame_count = 6, mono=to_gray) # mono = True 生成灰色背景

    def listadd(self,list1,list2):
        re_list = []
        for i in zip(list1,list2):
            re_list.append(i[0]+i[1])
        return re_list

    def do(self, cv_img=None, label_contents=None):
        if self.to_gray:
            cv_img = cv2.cvtColor(cv_img,cv2.COLOR_RGB2GRAY)
            cv_img = np.expand_dims(cv_img,-1)
        gen_img_h, gen_img_w = cv_img.shape[0], cv_img.shape[1]
        # gen_img_h, gen_img_w = 1000, 800
        # print('welcome do function............')
        bj_img = self.iter.get((gen_img_h, gen_img_w), scale_range=(0.5, 2), blur_range=(0, 1))
        if bj_img.shape != cv_img.shape:
            print(f'bj_img.shape:{bj_img.shape},cv_img.shape:{cv_img.shape}')
            bj_img = cv2.resize(bj_img,(gen_img_w,gen_img_h))
        ########### 在该区域做常规的背景融合 以及 使用背景做 padding 操作#############
        # step1 图像融合
        # bj_weight =  random.uniform(0.1, 0.2)
        bj_weight = random.uniform(0.1, 0.4)
        dst_img = cv2.addWeighted(cv_img, 1-bj_weight, bj_img, bj_weight, 0)
        if self.to_gray:
            dst_img = np.expand_dims(dst_img,-1)
        return dst_img, None

        # step2 给图像4个边的padding
        # bj_img_resize = cv2.resize(bj_img,(int(gen_img_w*1.2),int(gen_img_h*1.2)))
        # bj_img_resize_h, bj_img_resize_w = bj_img_resize.shape[0], bj_img_resize.shape[1]
        #
        # paste_x = random.randint(0,  bj_img_resize_w-gen_img_w)
        # paste_y = random.randint(0, bj_img_resize_h - gen_img_h)


        ###########  修改标签
        # new_boxes_list = []
        # if label_contents is not None:
        #     for label_content in label_contents:
        #         this_box_coord = ''
        #         tmp = label_content.split(',')
        #         cls = tmp[-1].strip()
        #         corrd_list = list(map(int, list(map(float, tmp[:-1]))))
        #
        #         adjust_coord = self.listadd(list1=[paste_x, paste_y] * (len(corrd_list) // 2), list2=corrd_list)
        #         for i in adjust_coord:
        #             this_box_coord += str(i)
        #             this_box_coord += ','
        #         this_box_coord += cls
        #         new_boxes_list.append(this_box_coord)
        #     bj_img_resize[paste_y:paste_y+gen_img_h,paste_x:paste_x+gen_img_w,:] = dst_img
        # return bj_img_resize,new_boxes_list



if __name__ == '__main__':
    def find_points_coord(list_):
        # ['1','2','3','4','5','6'] --> [(1,2),(3,4),(5,6)]
        # list_ 的长度 一定是偶数
        new_list = []
        for i in range(len(list_) // 2):
            new_list.append((int(float(list_[2 * i])), int(float(list_[2 * i + 1]))))
        return new_list

    img_path = r'./train_images'  # 图片的路径
    label_path = r'./train_gts'  # 标签的路径
    for j in os.listdir(img_path):
        file_name, extend_name = os.path.splitext(j)
        detail_img_path = os.path.join(img_path,j)
        detail_label_path = os.path.join(label_path,file_name + '.txt')

        img = cv2.imread(detail_img_path)


        with open(detail_label_path, 'r', encoding='utf-8') as reader:
            contents = reader.readlines()
        mix = My_mixup()
        deal_img, deal_label = mix.do(img, contents)
        for i in deal_label:
            tmp = i.split(',')
            cls = tmp[-1]
            points_list = find_points_coord(tmp[:-1])
            for points in zip(points_list[1:],points_list[:-1]):
                point1, point2 = tuple(list(points[0])), tuple(list(points[1]))
                cv2.line(deal_img,point1,point2,(0,255,0),2)
            cv2.line(deal_img, points_list[0], points_list[-1], (0, 255, 0), 2)


        cv2.imshow('1',deal_img)
        cv2.waitKey()






