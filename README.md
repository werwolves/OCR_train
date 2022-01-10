# OCR_project 单行文字识别

#### 
1. 支持 tensorflow 训练 densenet ocr模型，在配置文件中指定 "model_def_type": "tf.ocr_dense_net"，示例请参考
conf/conf.d/tf/fr/fr-num.json

```shell
python train.py --data_root=./data --config=tf/fr/fr-num

```


#### train:

```shell
python train.py --data_root=./data
```


#### test:

```shell
python train.py --data_root=./data
```


#################################### 训练步骤
ai-training使用方法简述：（以account为例）-->
1、设置字符集/conf/charset/account/account.txt下；
2、设置训练参数：conf/conf.d/tf/account/account.json；
3、启动训练sh trrain-tf-account.sh ;
4、生成的模型位于 output/tf/account下;
5、测试参考readme的启动命令

#################################### 补充
对于训练参数设置account-num.json中需要这种
charset_file字符集地址，
input_height/input_width/input_channels/训练样本规格，
train_args中的模型保存地址（model_path），
文件输入地址（input_path），
文件输入格式（默认folder-b64）,
是否随机加白边（rnd_proc_img）
以及其他的模型训练参数：batch_size,epochs,steps_per_epoch

