title: TPU中的指令并行和数据并行
date: 2019-08-03 01:56:17
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
---

>深度学习飞速发展过程中，人们发现原有的处理器无法满足神经网络这种特定的大量计算，大量的开始针对这一应用进行专用芯片的设计。谷歌的张量处理单元（Tensor Processing Unit，后文简称TPU）是完成较早，具有代表性的一类设计，基于脉动阵列设计的矩阵计算加速单元，可以很好的加速神经网络的计算。本系列文章将利用公开的TPU V1相关资料，对其进行一定的简化、推测和修改，来实际编写一个简单版本的谷歌TPU，以更确切的了解TPU的优势和局限性。


TPU V1定义了一套自己的指令集，虽然在介绍处理器时，往往会先谈指令集架构，但此处却把它放到了最后，这主要基于两个原因；其一在于个人的对处理器不太了解，这也是主要原因，其二在于公开资料中并没有TPU指令集的细节和TPU微架构的描述。从数据流和计算单元出发对TPU进行分析固然容易很多，但如果想理解TPU的设计思想，依旧需要回到其架构设计上进行分析。这一部分内容有些超出了我现有的能力，不当之处还请多多指正。

本文主要探讨从架构设计上看，TPU时如何做高性能和高效能的设计。高性能的多来自于并行，因此本文分别讨论了指令并行和数据并行的设计方法。由于论文中并未描述TPU指令集的具体设计，除特别说明外，本文关于TPU指令集的探讨均为推测；另外，SimpleTPU的指令设计并不系统/完整，此处仅阐明设计中的几种基本思想。

# 1. TPU的指令集
TPU的指令集采用CISC设计，共计有十多条指令，主要的五条指令包括
- Read_Host_Memory 将数据从CPU的内存中读取到TPU的Unified Buffer上
- Read_Weights 将weight从内存中读取到TPU的 Weight FIFO 上.
- MatrixMultiply/Convolve 执行卷积或矩阵乘法操作.
- Activate 执行人工神经网络中的非线性操作和Pooling操作（如有）
- Write_Host_Memory 将结果从Unified Buffer写回CPU内存.

从给出的五条指令可以看出，TPU的指令集设计和通用处理器有很大的不同。指令需要显示指定数据在内存和片上buffer之间搬移的过程。而执行指令（MatrixMultiply）直接指定了Buffer的地址，指令上并不能看到一系列通用寄存器。这是因为TPU本质上还是一个专用的处理芯片，其高性能和高效能都是建立在失去一定灵活性的前提下的。为了获得更高的性能，可以采用一系列的常规方法进行设计，包括

- 指令并行，即一次性处理更多指令，让所有执行单元高效运行
- 数据并行，即一次性处理多组数据，提高性能

后文会针对这两点做进一步描述，并简单讨论TPU设计中的更多其他的优化方法和方向。

# 2. 指令并行
## 2.1 Simple TPU中的流水线
为了提高吞吐率和时钟频率，处理器通常使用流水线设计，经典的五级流水线设计一般如下所示

  | | clk0 | clk1 |  clk2 |  clk3 |  clk4 |  clk5 |  clk6 |  clk7 | 
 --- | --- | --- | --- | --- | --- | --- | --- | ---
 instruction0 | IF | ID | EX | MEM | WB |  | | |
 instruction1 || IF | ID | EX | MEM | WB |  | |
 instruction2 ||| IF | ID | EX | MEM | WB |  | 
 instruction3 |||| IF | ID | EX | MEM | WB |  


其中，IF指取指(insturction fetch)，ID指指令译码（instruction decode），EX指执行（Execute），MEM指内存读写（Memory Access），WB指写回寄存器(Write back)。采用流水线设计可以提高性能，如果不采用流水线设计，那么instruction1需要在clk5才能开始进行IF，严重影响其性能；如果在同一周期完成IF/ID/EX/MEM/WB的功能，由于逻辑极其复杂，会严重影响工作频率。

TPU论文中介绍其采用四级流水线设计，Simple TPU中采用了两级流水线，来完成控制过程。

  | | clk0 | clk1 |  clk2 |  clk3 |  clk4 |  clk5 |  clk6 |  clk7 | 
 --- | --- | --- | --- | --- | --- | --- | --- | ---
 instruction0 | IF&ID | EX | MEM | WB |  | | |
 instruction1 || IF&ID | EX | MEM | WB |  | |
 instruction2 ||| IF&ID | EX | MEM | WB |  | 
 instruction3 |||| IF&ID | EX | MEM | WB |  

也认为Simple TPU内部有四级流水线，这是因为在实际执行过程中，包括了读取寄存器，执行和写回三个部分，这三个部分是流水设计的。

## 2.2 超长指令字（VLIW）
如前文所述，Simple TPU中有两个基本的计算单元——矩阵乘法阵列和池化计算单元。除此之外，还有一些没有显式描述的执行单元，譬如载入和存储。在这一前提下，即使TPU的指令流水线做得再好，每条指令占有的周期数也不可能小于1。如果其他执行单元的执行周期数很小，此时总会有一些执行单元处于闲置状态，处理器的瓶颈会出现在指令上。为了解决这一问题，很直接的想法时每个周期发射多条指令（另一个方法时让执行单元的执行时间变长，Simple TPU通过向量体系结构设计也有这一处理）。

由于TPU的专用性，以及计算过程中不存在跳转和控制的原因，采用VLIW设计多发射处理器似乎是一个很适合的方式。在这一设计下，指令发射结构时固定的，而且所有的冒险可以由编译器事先检测并处理，这很大程度可以降低硬件实现的复杂度。在Simple TPU中借鉴了VLIW的思想进行设计，如下所示(示意图)

![](https://raw.githubusercontent.com/cea-wind/blogs_pictures/master/img20190803171553.png)

其中各个字段具体描述如下
- model mask 指定了当前指令运行的模块
- load weight 指定了从内存将weight读取到SRAM的指令
- load act. & mac & store result 指定了将操作数（act.）读取到寄存器，乘加阵列计算以及将结果写回到存储器的过程
- set weight 指定了将操作数（weight）读取到计算阵列寄存器的过程
- load act. & pooling& store result field指定了将操作数（act.）读取到寄存器，完成pooling和归一化计算以及将结果写回到存储器的过程

VLIW的设计放弃了很多的灵活性和兼容性，同时将很多工作放到软件完成，但依旧适合在TPU这样的偏专用的处理器中使用。Simple TPU中没有对数据冲突、依赖进行任何处理，软件需要事先完成分析并进行规避。在这一设计下一条指令可以调度最多四个模块同时工作，效率得到了提升。

# 3. 卷积计算中的数据并行
## 3.1 单指令多数据（SIMD）

单指令多数据，故名思意是指在一条指令控制多组数据的计算。显然，TPU core的设计中采用了这样一种数据并行的方式——一条instruction控制了256*256个乘加计算单元（MatirxMultiply/Convolve）。根据指令流和数据流之间的对应关系，可以将处理器分为以下几个类别

- SISD，单指令流单数据流，顺序执行指令，处理数据，可以应用指令并行方法
- SIMD，单指令流多数据流，同一指令启动多组数据运算，可以用于开发数据级并行
- MISD，多指令流单数据流，暂无商业实现
- MIMD，多指令流多数据流，每个处理器用各种的指令对各自的数据进行操作，可以用在任务级并行上，也可用于数据级并行，比SIMD更灵活

由于TPU应用在规则的矩阵/卷积计算中，在单个处理器内部的设计上，SIMD是数据并行的最优选择。SIMD有多种实现方式，根据给出的描述（MatirxMultiply/Convolve指令接受B*256输入，输出B*256个结果），TPU中应该采用了类似向量体系结构的设计方法。

## 3.2 向量体系结构

如基本单元-矩阵乘法阵列所述，计算单元完成矩阵乘法计算，即向量计算。以《计算机体系结构 : 量化研究方法》给出的例子为例，如需计算
```
for(int i=0;i<N;i++)
    y[i] += a*x[i];
```
以MIPS为例，对于一般的标量处理器和向量处理器而言，执行的指令序列如下所示

![](https://raw.githubusercontent.com/cea-wind/blogs_pictures/master/img20190803171640.png)

对于卷积神经网络中的卷积操作而言，计算可以表示为（已忽略bias）
```
for(int i=0;i<M;i++){
    for(int j=0;j<N;j++){
        for(int k=0;k<K;k++){
            for(int c=0;c<C;c++){
                for(int kw=0;kw<KW;kw++){
                    for(int kh=0;kh<KH;kh++){
                        result(i,j,k) += feature(i+kw,j+kh,c)*w(k,kw,kh,c);
                    }
                }
            }
        }
    }
}
```
由于KW和KH可能为1（即卷积核的宽度和高度），而weight在计算过程中认为是固定在计算阵列内部的，因此调整循环顺序后有
```
for(int kw=0;kw<KW;kw++){
    for(int kh=0;kh<KH;kh++){
        for(int k=0;k<K;k++){
            for(int i=0;i<M;i++){
                for(int j=0;j<N;j++){
                    for(int c=0;c<C;c++){
                        result(i,j,k) += feature(i+kw,j+kh,c)*w(k,kw,kh,c);
                    }
                }
            }
        }
    }
}
```

其中第一二层循环通过指令进行控制，第三层循环在计算阵列中以256并行度进行计算，指令调度；第4-6层循环按向量处理器的设计思路进行设计，通过一条指令完成三层循环的计算。为了完成循环的计算，需要设置三个向量长度寄存器，另外，由于向量在SRAM中的地址并不连续，还需要设定三个不同的步幅寄存器。参考 基本单元-矩阵乘法阵列的代码，具体为
```
    short ubuf_raddr_step1;
    short ubuf_raddr_step2;
    short ubuf_raddr_step3;
    short ubuf_raddr_end1;
    short ubuf_raddr_end2;
    short ubuf_raddr_end3
```

 采用这样的设计，SimpleTPU中一条指令可以完成大量数据的计算，提高了数据并行度。这些数据会并行的进入到计算阵列中完成计算（可以认为是多条车道）。由于SimpleTPU中数据的读取延时是固定的（指从SRAM），因此向量化的设计较一般处理器还更为简单。

根据谷歌论文中的描述，TPU中有repeat fileld，但MatrixMultiply/Convolve指令长度有限，因此可能只有一组或两组向量长度寄存器和步幅寄存器，但设计思路应该类似。

# 4. 其他

从谷歌论文中的参数来看，TPU具有极高单位功耗下性能。这一部分来自于其内核设计，正如之前的文章中所描述的

- 采用了INT8数据类型进行计算
- 采用了脉动阵列优化计算
- 没有采用缓存，没有分支跳转，预测和数据冲突处理（编译器完成）

而从本文的内容可以看出，TPU还采用了简单的指令集设计+SIMD+向量体系结构+VLIW来进一步优化单位功耗下性能；除此之外，在V2/V3中google更进一步，还利用多核和多处理器设计进一步提高了性能。

 
# 参考
[1] Jouppi, Norman P. , et al. "In-Datacenter Performance Analysis of a Tensor Processing Unit." the 44th Annual International Symposium IEEE Computer Society, 2017.
[2] JohnL.Hennessy, and DavidA.Patterson. Computer architecture : a quantitative approach = 计算机体系结构 : 量化研究方法/ 5th ed.