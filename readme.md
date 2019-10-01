# 基于神经网络捏脸


## VERSION

```
1. Unity2019.2.1f1
2. python-2.7
3. tensorflow-1.1
4. dlib-19.18
5. numpy-1.15.4
6. opencv-contrib-python 3.4.0.12
```


## 论文

网易伏羲实验室、密歇根大学、北航和浙大的研究者提出了一种游戏角色自动创建方法，利用 Face-to-Parameter 的转换快速创建游戏角色，用户还可以自行基于模型结果再次进行修改，直到得到自己满意的人物。此项目按照[论文][i1]里的描述建立。

![](/image/t2.jpeg)



## 引擎预览

打开Unity, 点击菜单栏Tools->Preview, 通过此工具可以手动去捏脸。

![](/image/t1.jpg)


## 生成Database

打开Unity, 点击菜单栏Tools->GenerateDatabase

这里由引擎随机生成2000张图片， 其中80%用作训练集， 20%用作验证集。同时在图片同目录会生成二进制文件db_description，记录捏脸相关的参数。



## 脸部对齐 - Face alignment

对于输入图片，通过此工具dlib进行脸部截取。

```
pip install dlib
```

dlib 引用模型下载地址:

http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2 

http://dlib.net/files/dlib_face_recognition_resnet_model_v1.dat.bz2




[i1]: https://xueqiu.com/9217191040/133506937
[i2]: https://arxiv.org/abs/1909.01064
[i3]: http://www.sohu.com/a/339985351_823210
[i4]: https://blog.csdn.net/qiumokucao/article/details/81610628