title: Simple TPU的设计和性能评估
date: 2019-08-03 02:56:17
tags: 
  - Tensor Processing Unit
  - Architecture
  - Instruction Level Parallelism
  - Data Parallelism
  - VLIW
  - SMID
  - Vector Architecture
categories: 
  - TPU
author:
mathjax: true
---

>深度学习飞速发展过程中，人们发现原有的处理器无法满足神经网络这种特定的大量计算，大量的开始针对这一应用进行专用芯片的设计。谷歌的张量处理单元（Tensor Processing Unit，后文简称TPU）是完成较早，具有代表性的一类设计，基于脉动阵列设计的矩阵计算加速单元，可以很好的加速神经网络的计算。本系列文章将利用公开的TPU V1相关资料，对其进行一定的简化、推测和修改，来实际编写一个简单版本的谷歌TPU，以更确切的了解TPU的优势和局限性。

# 1. 完成SimpleTPU的设计
在[谷歌TPU概述和简化](https://cea-wind.github.io/2019/08/02/TPU_c1/)中给出过SimpleTPU的框图，如下图所示。

![](https://raw.githubusercontent.com/cea-wind/blogs_pictures/master/img20190803170350.png)

在[TPU中的脉动阵列及其实现](https://cea-wind.github.io/2019/08/02/TPU_c2/)中介绍了矩阵/卷积计算中的主要计算单元——乘加阵列（上图4），完成了该部分的硬件代码并进行了简单的验证；在[神经网络中的归一化和池化的硬件实现](https://cea-wind.github.io/2019/08/02/TPU_c3/)中介绍了卷积神经网络中的归一化和池化的实现方式（上图6），同时论述了浮点网络定点化的过程，并给出了Simple TPU中量化的实现方式，完成了该部分的硬件代码并进行了验证。

在[TPU中的指令并行和数据并行](https://cea-wind.github.io/2019/08/02/TPU_c4/)中对整个处理单元的体系结构进行了分析和论述，包括指令并行和数据并行两个方面。那么，如何在[TPU中的指令并行和数据并行](https://cea-wind.github.io/2019/08/02/TPU_c4/)中提到的设计思路下，将[TPU中的脉动阵列及其实现](https://cea-wind.github.io/2019/08/02/TPU_c2/)和[神经网络中的归一化和池化的硬件实现](https://cea-wind.github.io/2019/08/02/TPU_c3/)中提到的计算单元充分的利用，是完成Simple TPU设计的最后一步。根据SimpleTPU的框图可知，需要实现的功能包括

- 指令的取指和译码（上图4）
- Weight的读取（上图2）
- 各个执行单元的控制和调度（上图1）
- 读取图像和写回结果（上图5）
    
在SimpleTPU的设计中，指令的取指和译码和Weight的读取功能都较为简单，可直接参照代码。

在对各个执行单元进行控制和调度时需要确保各个单元可以共同执行，没有相互之间的数据依赖关系。

 除此之外，还需要单独实现读取图像和写回结果的功能。SimpleTPU中只关注核心的计算功能，该部分功能并未进行优化，后续对实现效果进行分析时，也会将该部分运行时间排除在外。

至此，Simple TPU的设计基本完成了，代码可参见[https://github.com/cea-wind/SimpleTPU](https://github.com/cea-wind/SimpleTPU)。

# 2. SimpleTPU的特性
SimpleTPU的主要特性包括
- 支持INT8乘法，支持INT32的累加操作
- 采用VLIW进行指令并行
- 采用向量体系结构进行数据并行

SimpleTPU依照Google TPU V1的设计思路，可以完成神经网络推理过程中的大部分运算。依据设计，支持的运算包括（理论）

Operate | Support
-|-
Conv3d | in_channels: Resource Constrained  <br> out_channels: Resource Constrained<br>kerner_size: Support<br>stride: support<br>padding: Support<br>dilation:Support<br>groups: Architecture Constrained<br>bias    :Support
ConvTranspose3d | The same as above
Maxpool2d | kernel_size: Support <br>stride: Support<br>padding: Support    
Avgpool2d | The same as above
Relu | Only support Relu as nonlinear function
BatchNorm2d | BatchNorm2d is merge with Conv or Pool when inference
Linear | Resource Constrained 
UpscalingNearest2D | Support (calling Avgpool2d multiple times.)
UpscalingBilinear2D | Support (calling Avgpool2d multiple times.)

其中，Resource Constrained代表该参数的取值范围有限，主要受限于SimpleTPU的存储设计等。由于架构设计上的问题，SimpleTPU对groupconv支持极为有限，在不合适的参数下效率可能远低于普通卷积；类似的，Google TPU也不能很好支持groupconv，并明确告知不制止depthwise conv（极度稀疏化的group conv）。

BatchNorm2d在推理过程中实际上时进行逐点的乘法和加法，其中加法计算可以融合到下一层或者上一层的卷积计算中进行，乘法计算可以和pooling计算融合。在SimpleTPU设计，每次完成卷积计算后，均要进行一次Pooling计算，即使网络中没有pooling层，SimipleTPU增加了一个1*1，stride=1的pooling层进行等价。

Upscaling操作通过pooling完成计算。这是因为在SimpleTPU中，reshape操作（支持的）是没有代价的。pooling操作可以完成双线性插值的计算，因此可以完成upscaling中的所有数值的计算。可以理解为通过pooling+reshape完成了upscaling的计算。

   
# 3. SimpleTPU的性能
Simple TPU设计了一个32×32的int8乘加阵列计算矩阵乘法和卷积，和一个1×32的int32乘法阵列进行池化和归一化的计算。根据Xilinx HLS工具的综合结果，在UltraScale+系列的FPGA器件上，工作频率可达500MHz。因此SimpleTPU的算力约为

$$32\times 32 \times 500MHz \times 2 = 1 Tops$$

作为对比，GoogleTPU V1的算力约为92Tops（int8），差异主要在SimpleTPU的规模为其1/64，同时在FPGA上的工作频率会低于ASIC的工作频率。

依据设计，SimpleTPU在适合的任务下会有很高的运行效率，TPU中的指令并行和数据并行中针对这一点又更为具体的描述。从宏观上看，SimpleTPU的各个运行单元可以流水并行的，即

![](https://raw.githubusercontent.com/cea-wind/blogs_pictures/master/img20190803172510.png)

而针对网络中计算量最大的全连接层和卷积层，针对性设计的乘法整列和向量计算的设计方法可以让其在每个时钟周期都完成有效的乘加计算；这意味着和CPU相比，SimpleTPU可以达到极高的效率。