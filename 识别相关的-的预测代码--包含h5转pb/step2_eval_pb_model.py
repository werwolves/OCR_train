# -*- coding:utf-8 -*-
import cv2,os,shutil,time
from model import Densenet
from image_utils import ImageRndUtils

alphabet_path='./args/num.txt'
model_path="./tensorflow_model/num.7.pb"
alphabet = open(alphabet_path, "r", encoding="utf-8").readline().strip()
densenet_ocr = Densenet(alphabet=alphabet, modelPath=model_path)


def find_file_name(undealed_file):
    image_name = ''
    dealed_file = undealed_file.split('.')
    if len(dealed_file) < 2:
        raise TypeError
    if len(dealed_file) == 2:
        image_name = dealed_file[0]
    elif len(dealed_file) > 2:
        for no, i in enumerate(dealed_file):
            if no == len(dealed_file) - 1:
                pass
            else:
                image_name = image_name + i + '.'
    if image_name.endswith('.'):
        image_name = image_name[:-1]
    return image_name


image_path = r'E:\PycharmProjects\need_updata_github\yolo3_yjk_pre\pic_cut'
label_path_1 = r'H:\zhonghang\ZH_YJK\new_test_data_label'

err_save = r'./err_file'
if not os.path.exists(err_save):
    os.mkdir(err_save)

def str_normal(conts):
    conts = conts.replace('.','').replace(',','')
    conts = conts.replace('-','').replace('(','').replace(')','')
    conts = conts.replace('¥','').replace('%','')
    conts = conts.strip()
    return conts

def fullocr(img):
    img = ImageRndUtils.updown_pad(img, (1, 1, 1, 1))
    pred_result = densenet_ocr.predict(img)
    return pred_result


acc_count = 0
err_count = 0
if __name__=='__main__':
    for file in os.listdir(image_path):
        if file.endswith('jpg'):
            tail_img_path = os.path.join(image_path,file)
            img=cv2.imread(tail_img_path,0)
            pred_result=fullocr(img)  # 1. 预测结果

            label_path = label_path_1 + '/' + find_file_name(file) + '.txt'

            try:
                with open(label_path, 'r', encoding='utf-8') as freader:
                    label_content = freader.readline()

                if pred_result.strip() == label_content.strip()[1:]:  # 因为label中有一个隐藏字符
                    acc_count = acc_count + 1
                else:
                    err_count = err_count + 1
                    print('err-pre:', pred_result)
                    print('err-label:', label_content)
                    detail_pic = err_save + '/' + i
                    cv2.imwrite(detail_pic, img)
                    detail_file = err_save + '/' + find_file_name(i) + '.txt'
                    with open(detail_file, 'w', encoding='utf-8') as writer:
                        writer.write('  pre:%s' % (pred_result))
                        writer.write('\n')
                        writer.write('label:%s' % (label_content))


            except Exception as e:
                print('exception:', e)


print('acc_count:',acc_count)
print('err_count:',err_count)
print('tot_acc:',float(acc_count/(acc_count+err_count)))


