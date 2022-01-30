#!/usr/bin/env python
# -*- coding: utf-8 -*-
import cv2,os,shutil,time
from densenet.model2 import Densenet
from image_utils import ImageRndUtils

alphabet_path='./args/full_7655.txt'
model_path="./args/full.7.h5"
alphabet = open(alphabet_path, "r", encoding="utf-8").readline().strip()
densenetconfig={
    'first_conv_filters':64,
    'first_conv_size':5,
    'first_conv_stride':2,
    'dense_block_layers':[8, 8, 8],
    'dense_block_growth_rate': 8,
    'trans_block_filters': 128,
    'first_pool_size':0,
    'first_pool_stride':2,
    'last_conv_size':0,
    'last_conv_filters':0,
    'last_pool_size':2,
}
imageconfig={
    'hight':32,
    'width':400,
    'channel':1,
}
densenet_ocr = Densenet(alphabet=alphabet, modelPath=model_path,imageconfig=imageconfig,densenetconfig=densenetconfig)

dir='../data/full_6422'
outdir='../error_full_6422'
if not os.path.exists(outdir):
    os.mkdir(outdir)

def str_normal(conts):
    '''
    消除中英文格式问题
    :param conts:
    :return:
    '''
    conts = conts.replace(' ', '').replace('（', '(').replace('）', ')')
    conts = conts.replace('：', ':').replace('一', '-').replace('—', '-')
    conts = conts.replace('，', ',').replace('“', '"').replace('”', '"')
    conts = conts.replace('【', '[').replace('】', ']')

    return conts

def fullocr(img,image_path=""):
    '''
    ocr识别
    :param img:
    :param image_path:
    :return:
    '''
    # img=ImageRndUtils.updown_pad(img, (4, 2, 4, 2))
    img = ImageRndUtils.updown_pad(img, (6, 2, 6, 2))
    pred_result = densenet_ocr.predict(img, image_path, merge_repeated=True)

    return pred_result

if __name__=='__main__':
    n,n1,n2=0,0,0
    t1=time.time()
    for file in os.listdir(dir):
        if not file.endswith('txt'):
            n=n+1
            # print(file)
            image_path=os.path.join(dir,file)
            txtpath = os.path.join(dir,file.split('.')[0] + '.txt')
            if not os.path.exists(txtpath):
                continue
            img=cv2.imread(image_path,0)
            pred_result=fullocr(img,image_path=image_path)

            f=open(txtpath,'r',encoding='utf-8')

            for conts in f.readlines():
                conts = conts.strip('\n')
                conts = str_normal(conts)
                pred_result = str_normal(pred_result)
                if conts != pred_result:
                    conts2=conts[1:]
                    if conts2 == pred_result:
                        continue
                    conts=conts.replace('0','O')
                    pred_result=pred_result.replace('0','O')
                    pred_result=pred_result.replace('|','')
                    if conts==pred_result:
                           n2=n2+1
                    else:
                        h, w = img.shape[:2]
                        print('predict %s\tPRED:%s\t\t%s\t\tLABEL:%s' % (file, pred_result, 'x', conts))

                        image_path2 = os.path.join(outdir, file)
                        txtpath2 = os.path.join(outdir, file.split('.')[0] + '.txt')
                        with open(txtpath2,'w+',encoding='utf-8') as fout:
                            fout.write("pred:"+pred_result+'  '+'lables:'+conts+'\n')
                        fout.close()
                        shutil.copyfile(image_path,image_path2)

                        n1 = n1 + 1

            f.close()
    t2=time.time()
    print('总耗时：{}'.format(t2-t1))
    print('平均耗时：{}'.format((t2-t1)/n))
    print('测试集数量：{}'.format(n))
    print('错误数量：{}'.format(n1))
    print('acc is {}'.format((n-n1)/n))
    print('消除0 和 O 以及 | 能修正的数量：{}'.format(n2))



