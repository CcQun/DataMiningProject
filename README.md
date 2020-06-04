## 数据集获取链接如下
链接：https://pan.baidu.com/s/1Nkzf019gcq9I2I7Dela8LA 
提取码：6dk7
## 数据集使用方法
将Data.zip解压到DataMiningProject文件夹下。其中txt文件夹下为原数据，包含正常状态数据和7种故障状态数据。通过TxtToCsv.py可以生成数据的csv格式文件。
## 项目结构介绍
#### 1.Code文件夹:保存代码文件
 - AETrain.py、AETest.py:自编码器实现
 - MLP.py:多层感知机实现
 - KNN.py:K近邻实现
 - TxtToCsv.py:将txt中的数据提取并转换为csv格式
 - DataSet.py:封装了提取数据的函数
#### 2.Data文件夹:保存数据集
#### 3.model文件夹:保存训练的模型