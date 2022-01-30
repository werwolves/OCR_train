# coding=utf-8

import os

import os.path as osp
import tensorflow as tf
from keras import backend as K

# from densenet import keys
from densenet.model2 import Densenet


def freeze_session(session, keep_var_names=None, output_names=None, clear_devices=True):
    """
    Freezes the state of a session into a prunned computation graph.

    Creates a new computation graph where variable nodes are replaced by
    constants taking their current value in the session. The new graph will be
    prunned so subgraphs that are not neccesary to compute the requested
    outputs are removed.
    @param session The TensorFlow session to be frozen.
    @param keep_var_names A list of variable names that should not be frozen,
                          or None to freeze all the variables in the graph.
    @param output_names Names of the relevant graph outputs.
    @param clear_devices Remove the device directives from the graph for better portability.
    @return The frozen graph definition.
    """
    from tensorflow.python.framework.graph_util import convert_variables_to_constants
    graph = session.graph
    with graph.as_default():
        freeze_var_names = list(set(v.op.name for v in tf.global_variables()).difference(keep_var_names or []))
        output_names = output_names or []
        output_names += [v.op.name for v in tf.global_variables()]
        input_graph_def = graph.as_graph_def()
        if clear_devices:
            for node in input_graph_def.node:
                node.device = ""
        frozen_graph = convert_variables_to_constants(session, input_graph_def,
                                                      output_names, freeze_var_names)
        return frozen_graph


input_fld = r"./"

weight_file = './args/num.7.h5'
alphabet_path='./args/num.txt'

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
    'hight':36,
    'width':285,
    'channel':1,
}

output_graph_name = os.path.splitext(os.path.basename(weight_file))[0] + ".pb"

output_fld = input_fld + 'tensorflow_model/'
if not os.path.isdir(output_fld):
    os.mkdir(output_fld)
weight_file_path = osp.join(input_fld, weight_file)

K.set_learning_phase(0)
densenet = Densenet(alphabet=alphabet, modelPath=weight_file,imageconfig=imageconfig,densenetconfig=densenetconfig)

net_model = densenet.basemodel
sess = densenet.sess

input_names = [input.op.name for input in net_model.inputs]
print("input names:", input_names)

output_names = [output.op.name for output in net_model.outputs]
print("output names:", output_names)

frozen_graph = freeze_session(sess, output_names=output_names)

from tensorflow.python.framework import graph_io

graph_io.write_graph(frozen_graph, output_fld, output_graph_name, as_text=False)

print('saved the constant graph (ready for inference) at: ', osp.join(output_fld, output_graph_name))