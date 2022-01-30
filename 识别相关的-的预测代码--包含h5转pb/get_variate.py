# -*- coding: utf-8 -*-

import tensorflow as tf

model_path = "./tensorflow_model/full.7.pb"

with tf.gfile.FastGFile(model_path,'rb')as f:
	graph_def = tf.GraphDef()
	graph_def.ParseFromString(f.read())
	tf.import_graph_def(graph_def,name='')

	tensor_name_list = [tensor.name for tensor in tf.get_default_graph().as_graph_def().node]
	for tensor_name in tensor_name_list:
		print(tensor_name, '\n')
	# with open('tensor_name_list_pb.txt','a')as t:
	# 	for tensor_name in tensor_name_list:
	# 		t.write(tensor_name+'\n')
	# 		print(tensor_name,'\n')
	# 	t.close()
	with tf.Session()as sess:
		op_list = sess.graph.get_operations()
		with open("var.txt",'w+')as f:
			for index,op in enumerate(op_list):
				f.write(str(op.name)+"\n")                   #张量的名称
				f.write(str(op.values())+"\n")              #张量的属