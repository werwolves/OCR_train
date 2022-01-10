# -*- coding: utf8 -*-
import time

from util import ArgUtils
from util.data_feeder import DataFeeder
from util.label_utils import Charset


class BaseModelDef:
    '''
    网络模型定义抽象类
    '''

    def __init__(self, name, **kwargs):
        self.name = name
        self.args = kwargs
        self._init_before(self.args)
        self.inputs = {}
        self.outputs = {}
        self.input = self.build_input_tensor()
        if not self.input is None:
            self.inputs['input'] = self.input

    def _init_before(self, args):
        self.input_height = args.get('input_height', 32)
        self.input_width = args.get('input_width', 360)
        self.input_channels = args.get('input_channels', 1)

    def is_single_channel(self):
        return self.input_channels == 1

    def data_shape(self):
        return [self.input_channels, self.input_height, self.input_width]

    def build_input_tensor(self):
        raise NotImplementedError()

    def build_model(self, input, **kwargs):
        raise NotImplementedError()

    def build_train_model(self, **kwargs):
        raise NotImplementedError()

    def build_infer_model(self, **kwargs):
        raise NotImplementedError()


class OcrModelDef:
    def init_ocr_model(self, **kwargs):
        self.charset = Charset(charset_file=kwargs.get('charset_file', None),
                               charset_text=kwargs.get('charset_text', None))

    def decode_tensor_to_labels(self, results):
        raise NotImplementedError()


class BaseTrainModel:
    '''
    模型训练抽象类
    '''

    def __init__(self, model_def, **kwargs):
        self.model_def = model_def
        self.args = kwargs
        self.epochs = self.args.get('epochs', 10)
        self.steps_per_epoch = self.args.get('steps_per_epoch', 1000)
        self.model_path = self.args.get('model_path', './output/model')
        self.use_gpu = self.args.get('use_gpu', False)
        self.gpu_memory = self.args.get('gpu_memory', 0)
        self.data_feeder_type = self.args.get('data_feeder_type', DataFeeder)
        self.data_feeder = None
        self.batch_size = self.args.get('batch_size',1)

    def init_model(self):
        '''
        初始化训练模型
        :return:
        '''
        raise NotImplementedError()

    def save_train_model(self, **kwargs):
        '''
        保存训练模型，以便增量训练
        :param kwargs:
        :return:
        '''
        pass

    def save_infer_model(self, **kwargs):
        '''
        保存预测模型
        :param kwargs:
        :return:
        '''
        pass

    def input_train_data_shape(self):
        return [-1] + self.model_def.data_shape()

    def load_data_feeder(self):
        '''
        加载训练集数据
        :return:
        '''
        self.data_feeder = self.data_feeder_type(self, is_test=False)


    def on_init(self):        # 引出网络模型
        self.load_data_feeder()
        self.init_model()     # 引出网络模型   将运行ocr/__init__.py 程序 step 5

    def load_last_train_model(self):
        pass

    def on_train_begin(self):
        self.load_last_train_model()

    def on_train_end(self, total_time_span):
        pass

    def run(self):
        self.on_init()    # step 3  在这里 引出网络
        begin_time = time.time()
        self.on_train_begin()    # 主要是为了加载 最后一次训练的模型，或是预训练模型
        self.do_run()
        self.on_train_end(time.time() - begin_time)

    def do_run(self):
        raise NotImplementedError()


class BaseInferModel:
    '''
    模型预测抽象类
    '''

    def __init__(self, model_def, **kwargs):
        self.model_def = model_def
        self.args = kwargs
        self.model_path = self.args.get('model_path', './output')
        self.use_gpu = self.args.get('use_gpu', False)
        self.gpu_memory = self.args.get('gpu_memory', 0)
        self.data_feeder_type = self.args.get('data_feeder_type', DataFeeder)
        self.data_feeder = None
        self.from_train_model = self.args.get('from_train_model', False)

    def load_infer_model(self):
        '''
        加载训练模型
        :return:
        '''
        raise NotImplementedError()

    def build_infer_model(self):
        raise NotImplementedError()

    def load_data_feeder(self):
        '''
        加载测试集数据
        :return:
        '''
        self.data_feeder = self.data_feeder_type(self, is_test=True)

    def on_init(self):
        self.load_data_feeder()
        if self.from_train_model:
            self.build_infer_model()
        else:
            self.load_infer_model()

    def on_test_begin(self):
        pass

    def on_test_end(self, total_time_span):
        pass

    def run(self):
        self.on_init()
        begin_time = time.time()
        self.on_test_begin()
        self.do_run()
        self.on_test_end(time.time() - begin_time)

    def do_run(self):
        raise NotImplementedError()


class ModelApp:
    def __init__(self, model_def_type, app_args, model_args, **kwargs):
        self.app_args = app_args
        self.model_args = model_args

        self.name = model_args.get('model_name', 'model')
        define_args = model_args.get('define_args')
        self.model_def = model_def_type(self.name, **define_args)   # 等于还要初始化 一下  -------------------

    def train(self, **kwargs):
        args = self.model_args.get('train_args', {}).copy()
        ArgUtils.extend_args(args, **self.app_args)
        ArgUtils.extend_args(args, **kwargs)
        #  train_model 本身是一个类
        train_model = self.model_def.build_train_model(**args)   #  CTC 类用于计算CTC的损失   self.model_def = TFOcrDenseNetModelDef
        #####################################################################  貌似在这里可以实施将某些网络给冻结的操作 ---- 2020-3-25
        # 方式1： 通过对网络层迭代的方式，对需要的网络层进行冻结操作 （把所有的网络层都给冻结掉，实验）
        # for layer in train_model.layers:
        #     layer.trainable = False
        # 方式2： 通过网络层的名字，对需要的网络层进行冻结操作
        # train_model.get_layer('block4_pool').trainable = False
        ######################################################################
        train_model.run()     # 引出网络模型 ---》 将直接运行 models/__init__.py 下的run程序  line122     step 2

    def test(self, **kwargs):
        args = self.model_args.get('test_args', {}).copy()
        ArgUtils.extend_args(args, **self.app_args)
        ArgUtils.extend_args(args, **kwargs)

        infer_model = self.model_def.build_infer_model(**args)
        infer_model.run()

    def run(self, mode, **kwargs):
        if mode == 'train':
            return self.train(**kwargs)       # 引出网络模型 step 1
        elif mode == 'test':
            return self.test(**kwargs)

        raise Exception("invalid mode")
