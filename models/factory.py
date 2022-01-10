# -*- coding: utf8 -*-
import json
import os
import re

from models import ModelApp
from util import Logger
from util.io_utils import FileUtils


class ModelFactory:
    def __init__(self, conf_path='./conf/conf.d'):
        self.conf_path = conf_path
        self.model_def_types = {}
        self.load_modules = set()

    def register_fluid_model_type(self):
        if 'fluid' in self.load_modules:
            return

        self.load_modules.add('fluid')
        from models.baidu.ocr_dense_net import FluidOcrDenseNetModelDef, FluidOcrDenseNetLstmModelDef
        from models.baidu.ocr_dpn import FluidOcrDpnModelDef
        from models.baidu.ocr_oline import FluidOcrOLineModelDef
        from models.baidu.table_cell_classify import TableCellClassifyModel
        from models.baidu.yolo import FluidYolo3ModelDef

        self.model_def_types['fluid.ocr_dense_net'] = FluidOcrDenseNetModelDef
        self.model_def_types['fluid.ocr_dense_net_lstm'] = FluidOcrDenseNetLstmModelDef
        self.model_def_types['fluid.ocr_dpn'] = FluidOcrDpnModelDef
        self.model_def_types['fluid.ocr_oline'] = FluidOcrOLineModelDef
        self.model_def_types['fluid.yolo3'] = FluidYolo3ModelDef
        self.model_def_types['fluid.table_cell_classify'] = TableCellClassifyModel

    def register_tf_model_type(self):
        if 'tf' in self.load_modules:
            return

        self.load_modules.add('tf')
        # from models.tf.nlp.text_classify_attention import TextClassifyAttention
        from models.tf.ocr.ocr_dense_net import TFOcrDenseNetModelDef
        from models.tf.ocr.ocr_dense_net import TFOcrDenseNetLstmModelDef
        # self.model_def_types['tf.text_classify_attention'] = TextClassifyAttention
        self.model_def_types['tf.ocr_dense_net'] = TFOcrDenseNetModelDef
        self.model_def_types['tf.ocr_dense_net_lstm'] = TFOcrDenseNetLstmModelDef

    def get_model_type(self, name):
        full_type = self.model_def_types.get(name, None)
        if not full_type is None:
            return full_type

        if name.startswith('fluid'):
            self.register_fluid_model_type()
        elif name.startswith('tf'):
            self.register_tf_model_type()   # 首先注册类 网络

        return self.model_def_types.get(name)  # 返回的是注册类（TFOcrDenseNetModelDef) 的实例

    def _process_text_vars(self, text, app_args):
        def replace_var(m):
            values = m.group(0)[2:-1].split(':')
            name = values[0]
            if name in app_args:
                return app_args[name]
            if len(values) > 0:
                return values[1].strip()
            raise ValueError('can not find the var:' + name)

        return re.sub('\${[^\}]+}', replace_var, text)

    def create_model_app(self, app_args):
        conf_name = app_args.config

        file = conf_name
        if not os.path.isfile(file):
            file = os.path.join(self.conf_path, conf_name + ".json")
            if not os.path.isfile(file):
                raise FileNotFoundError(file)

        app_args = app_args.__dict__.copy()
        Logger.info('load conf from :%s', file)
        text = FileUtils.read_text_from_file(file, strip=False)
        text = self._process_text_vars(text, app_args)

        app_args.pop('config')
        model_args = json.loads(text)
        model_def_type = self.get_model_type(model_args.pop('model_def_type'))   # model_def_type 为 TFOcrDenseNetModelDef 类的实例  这里面有注册的的模型的操作  （从配置的json文件中选取对应的模型）
        return ModelApp(model_def_type, app_args, model_args)
