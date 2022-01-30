# -*- coding: utf-8 -*-
from models.factory import ModelFactory
from util import BaseApp
# --config=tf/full_strs/full_strs --use_gpu=True --data_root=./data  # git 网页修改
# --config=tf/account/account-num --use_gpu=True --data_root=E:/PycharmProjects/work2/ai-training/data   训练官方的数字识别
# --config=tf/zch/zch-num --use_gpu=True --data_root=E:/PycharmProjects/work2/ai-training/data    训练注册号的识别
# python -u train.py --config=tf/full_strs/full_strs --use_gpu=True --data_root=/usr/hsc_projects/test_rec/ai-training/data_no_blank >train-11-17.log
'''
搜索: '卍' , 就可以找到修改过的可以识别空格的地方

'''

class TrainApp(BaseApp):
    def __init__(self):
        super().__init__()
        self.define_str_arg('config', '', 'config of model', required=True)
        self.define_str_arg('mode', 'train', 'run mode of mode')
        self.define_str_arg('data_root', './data', 'root path of data')
        self.define_str_arg('filter_train_name', None, 'input_path')
        self.define_str_arg('filter_test_name', None, 'input_path')
        self.define_str_arg('out_root', './output', 'output root path')
        self.define_int_arg('steps_per_epoch', None, 'steps_per_epoch')
        self.define_int_arg('batch_size', None, 'batch_size')
        self.define_bool_arg('use_gpu', False, 'use gpu')
        self.define_bool_arg('use_py_reader', None, 'use py_reader for fulid')
        self.define_bool_arg('attention', True, 'use gpu')
        self.define_str_arg('pre_weight', 'E:/PycharmProjects/work2/ai-training/pretraining_model/caibao.h5', 'pre train weight')
        self.factory = ModelFactory()

    def on_load(self):
        super().on_load()

    def do_run(self):   # run() 的实质性内容
        model_app = self.factory.create_model_app(self.args)    # 注册 2个识别模型
        model_app.run(self.args.mode)    # 引入网络模型 step 0


if __name__ == '__main__':
    ######################  加入 否则会报错 begin
    import tensorflow as tf
    from keras import backend as K

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    K.set_session(sess)
    ###################### 加入 否则会报错 end
    TrainApp().run()   # 执行子类---》BaseApp 的run方法

# with open('./read.txt') as read:
#     read.readline()
