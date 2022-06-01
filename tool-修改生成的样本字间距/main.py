import os
from tqdm import tqdm
import random

def is_chinese(ch):
    if u'\u4e00' <= ch <= u'\u9fff':
        return True
    else:
        return False



dirs_path = r'D:\my_datasets\train-title_realdata-2'

label_list = []
for i in tqdm(os.listdir(dirs_path)):
    if i.endswith('.txt'):
        detail_path = os.path.join(dirs_path, i)
        with open(detail_path, 'r', encoding='utf-8') as reader:
            content = reader.readline()
            label_list.append(content)
            # print(content)
label_list = list(set(label_list))

print(label_list)
print(len(label_list))

gen_txt_path = r'./gen_labels.txt'
with open(gen_txt_path, 'w', encoding='utf-8') as write:
        for label in tqdm(label_list):
            #################################
            new_label = ''
            if len(label) < 10:
                for str_ in label:
                    if is_chinese(str_): # 汉字
                        str_ =  str_ + ' ' * random.randint(1,3)
                    elif str_ == ' ':
                        str_ = ' ' * random.randint(1,3)
                    else:
                        pass
                    new_label += str_
            else:
                new_label = label
            new_label = new_label.lstrip().rstrip()
            #################################
            want_con = new_label + '\n'
            write.write(want_con)
