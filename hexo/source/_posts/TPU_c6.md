title: 神经网络加速器应用实例：图像分类
date: 2019-08-03 03:56:17
tags: 
  - Tensor Processing Unit
  - MLP
  - Image Classification
  - Hardware Accelarator  
categories: 
  - TPU
author:
---

>深度学习飞速发展过程中，人们发现原有的处理器无法满足神经网络这种特定的大量计算，大量的开始针对这一应用进行专用芯片的设计。谷歌的张量处理单元（Tensor Processing Unit，后文简称TPU）是完成较早，具有代表性的一类设计，基于脉动阵列设计的矩阵计算加速单元，可以很好的加速神经网络的计算。本系列文章将利用公开的TPU V1相关资料，对其进行一定的简化、推测和修改，来实际编写一个简单版本的谷歌TPU，以更确切的了解TPU的优势和局限性。
- [谷歌TPU概述和简化](https://cea-wind.github.io/2019/08/02/TPU_c1/)
- [TPU中的脉动阵列及其实现](https://cea-wind.github.io/2019/08/02/TPU_c2/)
- [神经网络中的归一化和池化的硬件实现](https://cea-wind.github.io/2019/08/02/TPU_c3/)
- [TPU中的指令并行和数据并行](https://cea-wind.github.io/2019/08/02/TPU_c4/)
- [Simple TPU的设计和性能评估](https://cea-wind.github.io/2019/08/02/TPU_c5/)
- [SimpleTPU实例：图像分类](https://cea-wind.github.io/2019/08/02/TPU_c6/)


# 1. 不仅仅是硬件的AI Inference
[Simple TPU的设计和性能评估](https://cea-wind.github.io/2019/08/02/TPU_c5/)中，一个神经网络加速器的硬件雏形已经搭建完成了；在[https://github.com/cea-wind/SimpleTPU](https://github.com/cea-wind/SimpleTPU) 上给出了相应的代码，和RTL仿真结果。在[TPU中的脉动阵列及其实现](https://cea-wind.github.io/2019/08/02/TPU_c2/)和[神经网络中的归一化和池化的硬件实现](https://cea-wind.github.io/2019/08/02/TPU_c3/)中，针对硬件实现中的关键模块也进行了仿真分析。但是，最终并没有给出一个可以实际运行的例子。这意味着，即使将这一部分代码应用到FPGA上，或者是实现在ASIC上后，也只有纸面性能却并不可用。

和很多其他的硬件设计不同，以Xilinx的AI Inference 解决方案为例（即之前的深鉴科技），用于AI Inference的设计需要考虑神经网络计算中的多样性，神经网络加速器是一个软件+硬件的解决方案。Xilinx叙述如下图[原始链接](https://www.xilinx.com/products/design-tools/ai-inference.html)。

![](https://raw.githubusercontent.com/cea-wind/blogs_pictures/master/img20190803161141.png)

从上往下看，这一套解决方案包括

- 主流的神经网络的框架的支持，包括caffe、Tensorflow和mxnet
- 提供模型压缩和优化的工具，以期在硬件上又更好的效能
- 提供模型量化的功能，使得浮点模型转化为定点模型
- 提供了Compiler，将模型映射为二进制指令序列
- 和Compiler相结合的Hardware

这意味着想真正使用之前设计的神经网络加速器——SimpleTPU，还需要软件的配合。即便模型压缩不在考虑范围内，也需要将模型量化为int8精度（SimpleTPU只支持int8乘法），同时利用Compiler生成指令序列。受限于个人能力，由于配套软件的缺失，下面的例子中的量化和指令均由手工生成。也正是由于这一原因，网络结构会尽可能简单，仅以保证本系列文章完整性为目的。

# 2. MLP分类实例

利用MLP对MNIST数据集进行手写数字分类的网络结构定义如下

```
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.hidden = nn.Linear(784,64)
        self.fc = nn.Linear(64,10)
 
    def forward(self, x):
        x = x.view(-1,784)
        x = self.hidden(x)
        x = self.fc(x)
        return F.log_softmax(x, dim=1)
```
生成指令后将其作为SimpleTPU的输入，并对其进行RTL仿真（testbench已经写好，直接运行即可），仿真结果如下图所示

![](https://raw.githubusercontent.com/cea-wind/blogs_pictures/master/img20190803161821.png)

前16张图的分类结果如下图所示

![](https://raw.githubusercontent.com/cea-wind/blogs_pictures/master/img20190803161851.png)

根据计算结果，可以分析得到其效率为84%。（去除了13K个用于读取图片和写回结果的时间，实际应用中，这一事件也会被计算时间覆盖）

LOC| Layers | Nonlinear function | Weights | Batch Size | % of Deployed
---|---|---|----|----|----
10 | 2 FC | Relu | 5M | 512 | 16%

作为参考，谷歌TPU中的数值为（尽管Simple TPU效率较高，但由于规模不同，无法直接对比效率；由于SimpleTPU完全按TPU设计，实际性能不可能高于TPU）

![](https://raw.githubusercontent.com/cea-wind/blogs_pictures/master/img20190803161808.png)

## 2.1 MLP运行分析
通过仿真波形，可以更直观的看出SimpleTPU的运行状态。下图中，读取Weight、乘加运算单元和Pooling共同工作可以反应TPU中的指令并行和数据并行中提到的指令并行。(由上倒下的ap_start分别是MXU，POOL，LOAD WEIGHT和INSTR FETCH的工作指示信号，同时拉高代表同时工作)

![](https://raw.githubusercontent.com/cea-wind/blogs_pictures/master/img20190803161933.png)

观察MXU内部的信号，可以看到计算过程中的数据并行（一条指令控制多组数据，且一个周期完成多组计算）。MXU每个周期都输出psum取值，一共有32个psum，计算一个psum需要32次乘加计算。

![](https://raw.githubusercontent.com/cea-wind/blogs_pictures/master/img20190803161950.png)

SimpleTPU为什么不够快（效率并没有接近100%）？这一问题可有下面的仿真波形看出（每次MXU启动前后都有若干个周期没有输出有效结果）

![](https://raw.githubusercontent.com/cea-wind/blogs_pictures/master/img20190803162002.png)

由于每次MXU执行一条向量计算指令会又若干个空闲的周期（超过64个周期，损失了10%以上的性能），导致了SimpleTPU在这一个网络上性能只有84%。MXU在启动之前需要32个周期来填满整个脉动阵列，而在输入结束后还需要32个周期来输出最后的结果。当采用HLS编写代码是，难以以这一控制力度来优化MXU的性能。如果采用Verilog HDL或者VHDL，可以采用指令之间的流水设计来消除这一延时。

# 3. CNN
由于手工对神经网络进行量化和layer间融合以及生成指令的复杂性，基于CNN的图像分类/分割网络的运行实例被无限期暂停了。

但是一个卷积计算的实例已经在[TPU中的脉动阵列及其实现](https://cea-wind.github.io/2019/08/02/TPU_c2/)中给出，证明了SimpleTPU计算卷积的能力。

根据[Simple TPU的设计和性能评估](https://cea-wind.github.io/2019/08/02/TPU_c5/)给出的特性，SimpleTPU可以高效支持绝大多数Operator，完成计算机视觉中的多种任务。当然，最大的缺陷在于SimpleTPU不显式支持ResNet，无法直接计算residual connection中的加法（可以进行channel concatenate之后再利用一次乘加计算间接支持resnet）。