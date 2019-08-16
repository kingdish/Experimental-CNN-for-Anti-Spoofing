# Experimental CNN for Anti-Spoofing

## 网络结构

所有网络均在`models`文件夹下的`mynet.py`内

`MyNetv1`：非常朴素的网络，代码比文字解释得更清楚

`MyNetv2`：

1. 输入RGB和IR图片，针对两者分别进行处理。conv->max_pool->conv->BN
2. 处理完后输入SPDBlock，训练时有相等概率输出(1)RGB, (2)IR, (3) RGB+IR
3. SPDBlock的输出再次conv->max_pool->conv->GAP->FC

`MyNetv3`：基于v1更改的依然朴素的网络，代码比文字更清楚

## 训练及测试

使用`trainer.py`对各个网络在`CIFAR-10`以及`CASIA-Webface`上进行pretraining，然后再在`CASIA-SURF`上训练。实验表明`MyNetv2Webface`的validation accuracy最高，约为91%。

## 使用

参考`inference.py`，默认网络的权重文件为`pretrained`文件夹内的`MyNetv2_WebFace_112_SURF.pth`。输入RGB和IR图像即可。

## 改进方向

- Residual
- SE block
- 在红外dataset上进行预训练