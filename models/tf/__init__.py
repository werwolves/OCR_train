import os

from models import BaseModelDef, BaseTrainModel, BaseInferModel, OcrModelDef
import tensorflow as tf
import numpy as np
from util import Logger
from util.io_utils import FileUtils

#######################
import pandas as pd
import matplotlib.pyplot as plt

def plot_history(history):
    hist = pd.DataFrame(history.history)
    hist['epoch'] = history.epoch

    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('train-loss')
    plt.plot(hist['epoch'], hist['loss'],
             label='Train loss')
    plt.plot(hist['epoch'], hist['acc'],
             label='Train acc')
    plt.ylim([0, 500])
    plt.legend()

    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('lr')
    plt.plot(hist['epoch'], hist['lr'],
             label='Train lr')
    plt.ylim([0, 2])
    plt.legend()
    plt.show()





class TFModelDef(BaseModelDef):

    def __init__(self, name, **kwargs):
        super().__init__(name, **kwargs)

    def build_input_tensor(self):
        return tf.keras.Input(self.data_shape(), name='input')

    def data_shape(self):
        return [self.input_height, self.input_width, self.input_channels]

    def build_infer_model(self, **kwargs):
        '''
        创建预测模型
        :param dir_name:
        :param file_name:
        :param use_gpu:
        :return:
        '''
        return TFInferModel(self, **kwargs)

    def convert_model_to_pb(self, model, save_pb_file):
        def freeze_session(session, keep_var_names=None, output_names=None, clear_devices=True):
            graph = session.graph
            with graph.as_default():
                freeze_var_names = list(set(v.op.name for v in tf.global_variables()).difference(keep_var_names or []))
                output_names = output_names or []
                # output_names += [v.op.name for v in tf.global_variables()]
                input_graph_def = graph.as_graph_def()
                input_graph_def = tf.graph_util.remove_training_nodes(input_graph_def)
                if clear_devices:
                    for node in input_graph_def.node:
                        node.device = ""

                frozen_graph = tf.graph_util.convert_variables_to_constants(session, input_graph_def,
                                                                            output_names, freeze_var_names)
                return frozen_graph

        folder, name = os.path.split(save_pb_file)

        sess = tf.keras.backend.get_session()

        # tf.saved_model.simple_save(
        #     sess,
        #     folder + '/1',
        #     inputs={t.name: t for t in model.inputs},
        #     outputs={t.name: t for t in model.outputs})

        output_names = [output.op.name for output in model.outputs]
        frozen_graph = freeze_session(sess, output_names=output_names)
        # # graph = sess.graph.as_graph_def()
        # # graph = tf.graph_util.remove_training_nodes(graph)
        # # frozen_graph = tf.graph_util.convert_variables_to_constants(sess,graph,output_names)
        tf.train.write_graph(frozen_graph, folder, name, as_text=False)
        #
        # # from tensorflow.python.framework import graph_io
        # # graph_io.write_graph(frozen_graph, folder, name, as_text=False)

        input_names = [input.op.name for input in model.inputs]
        Logger.info('saved pb model: %s, inputs=%s, outputs=%s' % (save_pb_file, input_names, output_names))


class TFOcrModelDef(TFModelDef, OcrModelDef):
    def __init__(self, name, **kwargs):
        super().__init__(name, **kwargs)
        self.init_ocr_model(**kwargs)

    def decode_tensor_to_labels(self, results):
        return []


class TFTrainModelCheckPoint(tf.keras.callbacks.Callback):
    def __init__(self, train_model):
        super().__init__()
        self.train_model = train_model

    def on_epoch_end(self, epoch, logs=None):
        self.train_model.on_epoch_end(epoch, logs=logs)


# noinspection PyAbstractClass
class TFTrainModel(BaseTrainModel):
    def __init__(self, model_def, **kwargs):
        super().__init__(model_def, **kwargs)
        self.log_step_period = self.args.get('log_step_period', 10)
        self.dir_mode = self.args.get('dir_mode', False)
        self.train_model = None
        self.base_model = None

    def on_init(self):
        if self.gpu_memory>0:
            self.init_session(self.gpu_memory)     # 建立会话
        super().on_init()  #   引出网络模型   #

    def init_session(self,gpu_fraction=0.5):
        num_threads = os.environ.get('OMP_NUM_THREADS')
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_fraction, allow_growth=False)

        if num_threads:
            sess = tf.Session(config=tf.ConfigProto(
                gpu_options=gpu_options,
                inter_op_parallelism_threads=num_threads,
                intra_op_parallelism_threads=num_threads, ))
        else:
            sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

        tf.keras.backend.set_session(sess)

    def save_train_model(self, **kwargs):
        save_to = kwargs.get('file_name') + ".h5"
        tf.keras.models.save_model(self.train_model, save_to, overwrite=True)

    def save_infer_model(self, **kwargs):
        '''
        保存预测模型
        :param dir_name:
        :param file_name:
        :param exe:
        :param main_program:
        :return:
        '''
        # print(kwargs)
        file_h5 = kwargs.get('file_name') + '.h5'
        # print(file_h5)
        self.base_model.save_weights(file_h5)


        # tf.keras.models.save_model(self.base_model, file_h5, overwrite=True,include_optimizer=True)

        # tf.keras.backend.set_learning_phase(1)
        # file_pb = kwargs.get('file_name') + '.pb'
        # self.model_def.convert_model_to_pb(self.base_model, file_pb)
        # tf.keras.backend.set_learning_phase(0)

    def get_feed_names(self, for_infer=True):
        return [self.model_def.input.name]

    def get_out_vars(self, for_infer=True):
        raise NotImplementedError()

    def on_epoch_begin(self, epoch, **kwargs):
        pass

    def on_epoch_end(self, epoch, **kwargs):
        # save_no = (epoch % 10) + 1   # hou
        save_no = epoch      # ----> hou

        save_train_path = os.path.join(self.model_path, "%s.%d.train" % (self.model_def.name, save_no))
        save_infer_path = os.path.join(self.model_path, "%s.%d" % (self.model_def.name, save_no))

        self.save_infer_model(file_name=save_infer_path)
        self.save_train_model(file_name=save_train_path)

    def try_load_pre_weight(self):
        """
        加载预训练模型
        :return:
        """
        pre_weight = self.args.get('pre_weight', None)
        pre_weight = False
        if not pre_weight or not os.path.exists(pre_weight):
            if pre_weight:
                Logger.warning('can not find the pre weight:%s!' % pre_weight)
            return False
        self.train_model.load_weights(pre_weight, by_name=True)
        print('load pre train weight from:%s' % pre_weight)
        return True

    def load_last_train_model(self):
        file_name = self.args.get('model_file_name', self.model_def.name)
        filter = '%s.*train%s' % (file_name, ".h5")
        file = FileUtils.get_file_by_max_write_time(self.model_path, filter)
        if file is None:
            return self.try_load_pre_weight()

        Logger.info('load last train model:%s' % file)
        self.train_model.load_weights(file)

        return True

    def on_train_begin(self):
        super().on_train_begin()

    def do_run(self):
        gen_train = self.data_feeder.gen_train_batch()

        has_val_data = self.data_feeder.has_val_batch_data()

        gen_val = self.data_feeder.gen_val_batch() if has_val_data else None
        train_result = self.train_model.fit_generator(gen_train,
                                       steps_per_epoch=self.steps_per_epoch,
                                       validation_data=gen_val,
                                       validation_steps=self.steps_per_epoch,
                                       epochs=self.epochs,
                                       callbacks=self.get_train_callbacks(),
                                       shuffle = True)
        print('train-info:', train_result.history.keys())
        print('train-info-:', train_result.history.values())
        plot_history(train_result)
    # def get_train_callbacks(self): #  没有修改前的
    #     lr_schedule = lambda epoch: 0.001 * 0.4 ** epoch
    #
    #     #lr_schedule = lambda epoch: 0.0005 * 0.4 ** epoch
    #
    #     learning_rate = np.array([lr_schedule(i) for i in range(20)])
    #     change_lr = tf.keras.callbacks.LearningRateScheduler(lambda epoch: float(learning_rate[epoch]))
    #
    #     return [TFTrainModelCheckPoint(self), change_lr]

    def get_train_callbacks(self):  #  change lr
        # lr_schedule = lambda epoch: 0.001 * 0.4 ** epoch

        #lr_schedule = lambda epoch: 0.0005 * 0.4 ** epoch

        # learning_rate = np.array([lr_schedule(i) for i in range(20)])
        # change_lr = tf.keras.callbacks.ReduceLROnPlateau(lambda epoch: float(learning_rate[epoch]))
        change_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='train_loss', factor=0.1, min_lr=1e-10, patience=1, verbose=1) # hou
        return [TFTrainModelCheckPoint(self), change_lr]


    def do_train_batch(self, **kwargs):
        pass


class TFInferModel(BaseInferModel):
    def __init__(self, model_def, **kwargs):
        super().__init__(model_def, **kwargs)
        self.model_def = model_def
        self.sess = None
        self.outputs = []
        self.input_map = {}

    def get_feed_names(self):
        return ['input:0']

    def get_out_names(self):
        raise NotImplementedError()

    def get_model_file_for_load(self, ext_name='.pb'):
        file_name = self.args.get('model_file_name', None)
        if file_name is None:
            filter = "%s.*%s" % (self.model_def.name, ext_name)
            model_file = FileUtils.get_file_by_max_write_time(self.model_path, filter)
            if model_file is None:
                raise ValueError("can not find the model file in path:%s/%s" % (self.model_path, filter))
            return model_file
        else:
            return file_name

    def load_graph(self, frozen_graph_filename):
        # We load the protobuf file from the disk and parse it to retrieve the
        # unserialized graph_def
        with tf.gfile.GFile(frozen_graph_filename, "rb") as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())

            # Then, we import the graph_def into a new Graph and returns it
            with tf.Graph().as_default() as graph:
                # The name var will prefix every op/nodes in your graph
                # Since we load everything in a new graph, this is not needed
                tf.import_graph_def(graph_def, name='')

        return graph

    def load_pb_model(self, pb_file):
        assert os.path.exists(pb_file)
        graph = self.load_graph(pb_file)

        config = tf.ConfigProto(allow_soft_placement=True)
        if self.gpu_memory>0:
            config.gpu_options.per_process_gpu_memory_fraction = self.gpu_memory

        config.gpu_options.allow_growth = True

        for name in self.get_feed_names():
            tensor = graph.get_tensor_by_name(name)
            names = name.split(':')
            self.input_map[names[0]] = tensor

        self.outputs = []
        for name in self.get_out_names():
            tensor = graph.get_tensor_by_name(name)
            self.outputs.append(tensor)

        if len(self.outputs) == 1:
            self.outputs = self.outputs[0]

        with graph.as_default():
            self.sess = tf.Session(graph=graph, config=config)

    def load_infer_model(self):
        pb_file = self.get_model_file_for_load()
        self.load_pb_model(pb_file)

    def do_run(self):
        for data in self.data_feeder.gen_test_batch():
            self.do_test_batch(data)

    def do_test_batch(self, data):
        raise NotImplementedError()
