title: 动手写一个简单版的谷歌TPU
tags:
  - Neural Network
  - Tensor Processing Unit
categories:
  - TPU
date: 2019-08-02 22:34:00
---
>深度学习飞速发展过程中，人们发现原有的处理器无法满足神经网络这种特定的大量计算，大量的开始针对这一应用进行专用芯片的设计。谷歌的张量处理单元（Tensor Processing Unit，后文简称TPU）是完成较早，具有代表性的一类设计，基于脉动阵列设计的矩阵计算加速单元，可以很好的加速神经网络的计算。本系列文章将利用公开的TPU V1相关资料，对其进行一定的简化、推测和修改，来实际编写一个简单版本的谷歌TPU，以更确切的了解TPU的优势和局限性。


# 1. TPU设计分析

人工神经网络中的大量乘加计算（譬如三维卷积计算）大多都可以归纳成为矩阵计算。而之前有的各类处理器，在其硬件底层完成的是一个（或多个）标量/向量计算，这些处理器并没有充分利用矩阵计算中的数据复用；而Google TPU V1则是专门针对矩阵计算设计的功能强大的处理单元。参考Google公开的论文[In-Datacenter Performance Analysis of a Tensor Processing Unit]( https://arxiv.org/ftp/arxiv/papers/1704/1704.04760.pdf )，TPU V1的结构框图如下所示

![](https://raw.githubusercontent.com/cea-wind/blogs_pictures/master/img20190803025510.png)

结构框图中最受瞩目的是巨大的Matrix Multiply Unit，共计64K的MAC可以在700MHz的工作频率下提供92T int8 Ops的性能。这样一个阵列进行矩阵计算的细节将会在基本单元-矩阵乘法阵列进行更进一步的阐述。TPU的设计关键在于充分利用这一乘加阵列，使其利用率尽可能高。

结构图中其他的部分基本都是为尽可能跑满这个矩计算阵列服务的，据此有以下设计
- **Unified Buffer 提供了256×8b@700MHz的带宽**（即167GiB/s，0.25Kib×700/1024/1024=167GiB/s），以保证计算单元不会因为缺少Data in而闲置；
- **Local Unified Buffer 的空间高达24MiB**，这意味着计算过程的中间结果几乎无需和外界进行交互，也就不存在因为数据带宽而限制计算能力的情况；
- Matrix Multiply Unit中**每个MAC内置两个寄存器存储Weight**，当一个进行计算时另一个进行新Weight的载入，以掩盖载入Weight的时间；
- 30GiB/s的带宽完成256×256Weight的载入需要大约**1430个Cycles**，也就意味着一组Weight至少需要计算1430Cycles，因此Accumulators的**深度需要为2K**（1430取2的幂次，论文中给出的数值是1350，差异未知）；
- 由于MAC和Activation模块之间需要同时进行计算，因此Accumulators需要用两倍存储来进行pingpang设计，因此**Accumulators中存储的深度设计为4k**；

因此从硬件设计上来看，只要TPU ops/Weight Byte达到1400左右，理论上TPU就能以接近100%的效率进行计算。但在实际运行过程中，访存和计算之间的调度，读写之间的依赖关系（譬如Read After Write，需要等写完才能读），指令之间的流水线和空闲周期的处理都会在一定程度影响实际的性能。为此，TPU设计了一组指令来控制其访问存和计算，主要的指令包括

- Read_Host_Memory
- Read_Weights
- MatrixMultiply/Convolve
- Activation
- Write_Host_Memory

所有的设计都是为了让矩阵计算单元一直处于工作状态，即希望所有其他指令可以被MatrixMultiply指令所掩盖，因此TPU采用了分离数据获取和执行的设计（Decoupled-access/execute），这意味着在发出Read_Weights指令之后，MatrixMultiply就可以开始执行，不需要等待Read_Weight指令完成；如果Weight/Activation没有准备好，matrix unit会停止。

需要注意的是，一条指令可以执行数千个周期，因此TPU设计过程中没有对流水线之间的空闲周期进行掩盖(存疑)，这是因为由Pipline带来的数十个周期的浪费对最终性能的影响不到1%。

关于指令的细节依旧不是特别清楚，更多细节有待讨论补充。

# 2. TPU的简化
实现一个完整的TPU有些过于复杂了，为了降低工作量、提高可行性，需要对TPU进行一系列的简化；为做区分，后文将简化后的TPU称为SimpleTPU。所有的简化应不失TPU本身的设计理念。

TPU中为了进行数据交互，存在包括PCIE Interface、DDR Interface在内的各类硬件接口；此处并不考虑这些标准硬件接口的设计，各类数据交互均通过AXI接口完成；仅关心TPU内部计算的实现，更准确的来说，Simple TPU计划实现TPU core，即下图红框所示。

![](https://raw.githubusercontent.com/cea-wind/blogs_pictures/master/img20190803025608.png)

由于TPU的规模太大，乘法器阵列大小为256×256，这会给调试和综合带来极大的困难，因此此处将其矩阵乘法单元修改为32×32，其余数据位宽也进行相应修改，此类修改包括

Resource | TPU | SimpleTPU
--- | --- | ---
Matrix Multiply Unit | 256×256 | 32×32
Accumulators RAM | 4K×256×32b | 4K×32×32b
Unified Buffer | 96K×256×8b | 16K×32×8b
    
- 由于Weight FIFO实现上的困难（难以采用C语言描述）, Weight采用1K*32*8b的BRAM存放，Pingpang使用；
- 由于Matrix Multiply Unit和Accumulators之间的高度相关性，SimpleTPU将其合二为一了；
- 由于Activation和Normalized/Pool之间的高度相关性，SimpleTPU将其合二为一了（TPU本身可能也是这样做的），同时只支持RELU激活函数；
- 由于并不清楚Systolic Data Setup模块到底进行了什么操作，SimpleTPU将其删除了；SimpleTPU采用了另一种灵活而又简单的方式，即通过地址上的设计，来完成卷积计算；
- 由于中间结果和片外缓存交互会增加instruction生成的困难，此处认为计算过程中无需访问片外缓存；(这也符合TPU本身的设计思路，但由于Unified Buffer大小变成了1/24，在这一约束下只能够运行更小的模型了)
- 由于TPU V1并没有提供关于ResNet中加法操作的具体实现方式，SimpleTPU也不支持ResNet相关运算，但可以支持channel concate操作；（虽然有多种方式实现Residual Connection，但均需添加额外逻辑，似乎都会破坏原有的结构）

简化后的框图如下所示，模块基本保持一致

![](https://raw.githubusercontent.com/cea-wind/blogs_pictures/master/img20190803170350.png)

# 3. 基于Xilinx HLS的实现方案
一般来说，芯片开发过程中多采用硬件描述语言（Hardware Description Language, HDL），譬如Verilog HDL或者VHDL进行开发和验证。但为了提高编码的效率，同时使得代码更为易懂，SimpleTPU试图采用C语言对硬件底层进行描述；并通过HLS技术将C代码翻译为HDL代码。由于之前使用过Xilinx HLS工具，因此此处依旧采用Xilinx HLS进行开发；关于Xilinx HLS的相关信息，可以参考高层次综合（HLS）-简介，以及一个简单的开发实例利用Xilinx HLS实现LDPC译码器。

虽然此处选择了Xilinx HLS工具，但据我所了解，HLS可能并不适合完成这种较为复杂的IP设计。尽管SimpleTPU已经足够简单，但依旧无法在一个函数中完成所有功能，而HLS并不具有函数间相对复杂的描述能力，两个模块之间往往只能是调用关系或者通过FIFO Channel相连。但由于HLS具有**易写、易读、易验证**的有点，此处依旧选择了HLS作为开发语言，并通过一些手段规避掉了部分问题。真实应用中，采用HDL或者HDL结合HLS进行开发是更为合适的选择。

按规划之后将给出两个关键计算单元的实现，以及控制逻辑和指令的设计方法；最后将给出一个实际的神经网络及其仿真结果和分析。具体包括
- [谷歌TPU概述和简化](https://cea-wind.github.io/2019/08/02/TPU_c1/)
- [TPU中的脉动阵列及其实现](https://cea-wind.github.io/2019/08/02/TPU_c2/)
- [神经网络中的归一化和池化的硬件实现](https://cea-wind.github.io/2019/08/02/TPU_c3/)
- [TPU中的指令并行和数据并行](https://cea-wind.github.io/2019/08/02/TPU_c4/)
- [Simple TPU的设计和性能评估](https://cea-wind.github.io/2019/08/02/TPU_c5/)
- [SimpleTPU实例：图像分类](https://cea-wind.github.io/2019/08/02/TPU_c6/)