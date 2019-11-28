# 基于神经网络捏脸


## Reinferences

```
1.  Unity-2019.2.1f1
2.  python-3.5
3.  dlib-19.18
4.  numpy-1.15.4
5.  torch-1.1.0
6.  opencv-contrib-python 3.4.0.12
7.  tqdm-4.23.4
8.  argparse-1.4.0
9.  scipy-1.0.1
10. tensorboardX
```


## 论文

网易的研究者提出了一种游戏角色自动创建方法，利用 Face-to-Parameter 的转换快速创建游戏角色，用户还可以自行基于模型结果再次进行修改，直到得到自己满意的人物。此项目按照[论文][i2]里的描述建立。


[Face-to-ParameterTranslationforGameCharacterAuto-Creation][i2]


![](/image/t2.jpeg)


## 引擎预览

打开Unity, 点击菜单栏Tools->Preview, 通过此工具可以手动去捏脸。当然， 此项目是通过神经网络生成参数params。

![](/image/t1.jpg)


## Database

打开Unity, 点击菜单栏Tools->GenerateDatabase

![](/image/t6.jpg)

点击generate按钮之后，在export目录会生成两个文件夹trainset和testset， 分别用作训练集和测试集。 勾选trainset noise之后，trainset随机生成的噪点大概是1/95。

这里引擎最大随机生成10000张图片， 其中80%用作训练集。同时在图片同目录会生成二进制文件db_description，记录捏脸相关的参数，作为imitator输入的参数。

生成图片分辨率：512x512,  不同的是使用Unity引擎代替论文里的Justice引擎。

![](/image/t4.jpg)

## 脸部对齐 - Face alignment

对于输入图片，通过此工具dlib进行脸部截取。

```
pip3 install dlib
```

或者使用conda安装

```
conda install -c menpo dlib
```

dlib 引用模型下载地址:

http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2 

http://dlib.net/files/dlib_face_recognition_resnet_model_v1.dat.bz2

##  LightCNN

[light_cnn][i5]出自2016 cvpr吴翔A Light CNN for Deep Face Representation with Noisy Labels，light_cnn优势在于一个很小的模型和一个非常不错的识别率。论文里使用LightCNN用于生成256 demonsion 信息，进而得到Loss函数L1, 即Discriminative Loss, 衡量引擎生成的图片和Imitator生成的图片差异。


训练好的模型下载连接：

google driver: [LightCNN-29 v2][i6]

baidu  yun:	   [LightCNN-29 v2][i7]


## 人脸分割


论文里使用人脸分割，提取局部面部特征， 从而计算Facial Content Loss， 下面列出了我网上找到相关的人脸分割的相关介绍和数据集。

1. 介绍

	
	[helen dataset 介绍]: http://www.ifp.illinois.edu/~vuongle2/helen

	[Exemplar-Based Face Parsing]: http://pages.cs.wisc.edu/~lizhang/projects/face-parsing/

	图像解析与编辑[中国科学院信息工程研究所网络空间技术实验室ppt]: https://pan.baidu.com/s/1FYznfGG914pPaU5bs0-4dw

2. 数据集

	helen_small4seg

	https://share.weiyun.com/5Q9ST03 密码：ndks4g



不同于论文里使用的resnet50，此项目引用的模型是BiSeNet

预训练model:	https://pan.baidu.com/s/1AEc7CJGirsdxOouD3boRBQ  

下载后存放在face-parsing.PyTorch/res/cp 目录下


最后的效果如图：

![](/image/t3.jpg)



## Operation


### 训练 train

进入git 下载目录， 按照下面命令就可以train， 训练的时候每隔100步会生成preview 图片，实时查看训练结果; 每隔500步就会保存下当前模型, 保存在model文件夹下

```sh
cd /path/to/workdir

cd neural/

mkdir dat/

# 这里需要将下载的lightcnn, dlib等模型拷贝过来
copy yours_download_model_path data/

python3 main.py \
	--phase=train_imitator \
	--batch_size=4 \
	--learning_rate=0.01 \
	--total_steps=30000	\
	--prev_freq=100	\
	--save_freq=500	\
	--path_to_dataset="../export/trainset_female/"

# 开启tensorboard 查看graph运行情况
tensorboard --logdir logs

```

最后训练得到的效果如下图：

![](/image/t7.jpg)

(上图是左上角是参考图，右上角是imitator生成图，左下角是生成图的脸部语义切割图， 右下角是部位边缘图)

### reinference

如果因为中间因为偶然因素（比如说断电）退出了训练， 又想从之前保存的model中恢复出来， 这时候可以先把model里的模型文件copy到reinference目录， 然后执行下面命令：

```sh
python3 main.py \
	--phase=inference_imitator	\
	--total_steps=30000	\
	--path_to_dataset="../export/trainset_female/" \
	--imitator_model="model_imitator_100000.pth"
```


##### imitator pretrained model:


https://pan.baidu.com/s/1qDRPAzn3m9VxX2Z-oENBWQ

train中的效果参考视频：

 https://youtu.be/FKJ-rvXS39U

 https://www.bilibili.com/video/av76020308/

### preview

效果预览, 生成捏脸参数二进制文件

```sh
python3 main.py \
	--phase=evaluate \
	--total_eval_steps=1000 \
	--imitator_model="model_imitator_100000.pth" \
	--eval_image='../export/testset_female/db_0000_4.jpg'
```

运行脚本后， 会在output/eval目录生成一个eval.bytes, 然后打开unity, 菜单栏tools->SelectModel, 即可在引擎预览到效果


[i1]: https://xueqiu.com/9217191040/133506937
[i2]: https://arxiv.org/abs/1909.01064
[i3]: http://www.sohu.com/a/339985351_823210
[i4]: https://blog.csdn.net/qiumokucao/article/details/81610628
[i5]: https://github.com/AlfredXiangWu/LightCNN
[i6]: https://drive.google.com/open?id=1Jn6aXtQ84WY-7J3Tpr2_j6sX0ch9yucS
[i7]: https://pan.baidu.com/s/1E_rGkbqzf0ppyl5ks9FSLQ
[i8]: https://github.com/zllrunning/face-parsing.PyTorch