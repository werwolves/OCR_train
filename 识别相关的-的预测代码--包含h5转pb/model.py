#_*_coding:utf-8_*_
#浪淘沙
import cv2
import  tensorflow as tf
import numpy as np

class Densenet:
    def __init__(self,alphabet, modelPath):
        self.characters = alphabet + u' '
        self.nclass=len(self.characters)
        self.input_height = 36
        self.config = tf.ConfigProto(allow_soft_placement=True)
        self.config.gpu_options.per_process_gpu_memory_fraction = 0.3
        self.config.gpu_options.allow_growth = True
        self.graph = self.load_graph(modelPath)
        with  self.graph.as_default():
            self.sess = tf.Session(graph=self.graph, config=self.config)
        print([n.name for n in self.sess.graph.as_graph_def().node])
        self.input = self.sess.graph.get_tensor_by_name('the_input:0')
        self.input_length=self.sess.graph.get_tensor_by_name('input_length:0')
        self.out = self.sess.graph.get_tensor_by_name('out_idxes/SparseToDense:0')

    def load_graph(self,pbpath):
        with tf.gfile.GFile(pbpath, "rb") as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
        with tf.Graph().as_default() as graph:
            tf.import_graph_def(graph_def, name='')
        return graph

    def pad_img(self,im, padding, fill_value=(255, 255)):

        t, r, b, l = padding
        if len(im.shape) == 3:
            im = np.pad(im, ((t, b), (l, r), (0, 0)), mode='constant', constant_values=fill_value)
        else:
            im = np.pad(im, ((t, b), (l, r)), mode='constant',
                        constant_values=fill_value)
        return im

    def predict(self, img):
        height, width = img.shape[:2]
        scale = height * 1.0 / self.input_height
        width = int(width / scale)
        if width<9:
            return ''
        img = cv2.resize(img, (width, self.input_height))
        img = img.astype(np.float32) / 255.0 - 0.5
        input_l=np.zeros((1, 1), dtype=np.int64)
        input_l[0] = width // 8

        # X = [img.reshape([1, 32, width, 1]),input_l]
        X = img.reshape([1, self.input_height, width, 1])
        y_pred = self.sess.run(self.out, feed_dict={self.input: X,
                                                    self.input_length:input_l})
        out = self.id_2_v(y_pred[0])

        return out

    def id_2_v(self, vec):
        '''
        向量转文本
        :param vec:
        :return:
        '''
        text = ""
        for y in vec:
            i = int(y)
            if i >= 0 and i < self.nclass:
                text += self.characters[i]
        return text.strip()

