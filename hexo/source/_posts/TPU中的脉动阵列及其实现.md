title: TPU中的脉动阵列及其实现
tags: []
categories: []
date: 2019-08-03 00:17:00
author:
---
深度学习飞速发展过程中，人们发现原有的处理器无法满足神经网络这种特定的大量计算，大量的开始针对这一应用进行专用芯片的设计。谷歌的张量处理单元（Tensor Processing Unit，后文简称TPU）是完成较早，具有代表性的一类设计，基于脉动阵列设计的矩阵计算加速单元，可以很好的加速神经网络的计算。本系列文章将利用公开的TPU V1相关资料，对其进行一定的简化、推测和修改，来实际编写一个简单版本的谷歌TPU，以更确切的了解TPU的优势和局限性。

动手写一个简单版的谷歌TPU系列目录
    谷歌TPU概述和简化

    TPU中的脉动阵列及其实现

    神经网络中的归一化和池化的硬件实现

    TPU中的指令并行和数据并行

    Simple TPU的设计和性能评估

    SimpleTPU实例：图像分类

    拓展

    TPU的边界（规划中）

    重新审视深度神经网络中的并行（规划中）

 

    本文将对TPU中的矩阵计算单元进行分析，并给出了SimpleTPU中32×32的脉动阵列的实现方式和采用该阵列进行卷积计算的方法，以及一个卷积的设计实例，验证了其正确性。代码地址https://github.com/cea-wind/SimpleTPU/tree/master/lab1

1. 脉动阵列和矩阵计算
    脉动阵列是一种复用输入数据的设计，对于TPU中的二维脉动阵列，很多文章中构造了脉动阵列的寄存器模型，导致阅读较为困难，而实际上TPU中的二维脉动阵列设计思路十分直接。譬如当使用4×4的脉动阵列计算4×4的矩阵乘法时，有

![upload successful](\images\pasted-3.png)

![upload successful](\\images\pasted-4.png\)

![upload successful](\\images\pasted-5.png\)

    如上图所示，右侧是一个乘加单元的内部结构，其内部有一个寄存器，在TPU内对应存储Weight，此处存储矩阵B。左图是一个4×4的乘加阵列，假设矩阵B已经被加载到乘加阵列内部；显然，乘加阵列中每一列计算四个数的乘法并将其加在一起，即得到矩阵乘法的一个输出结果。依次输入矩阵A的四行，可以得到矩阵乘法的结果。

    由于硬件上的限制，需要对传播路径上添加寄存器，而添加寄存器相对于在第i个时刻处理的内容变成了i+1时刻处理；这一过程可以进行计算结果上的等效。如下图所示，采用z-1代表添加一个延时为1的寄存器，如果在纵向的psum传递路径上添加寄存器，为了保证结果正确，需要在横向的输入端也添加一个寄存器（即原本在i进行乘加计算的两个数均在i+1时刻进行计算）。给纵向每个psum路径添加寄存器后，输入端处理如右图所示。（下图仅考虑第一列的处理）

clip_image002[5]

    当在横向的数据路径上添加寄存器时，只要每一列都添加相同延时，那么计算结果会是正确的，但是结果会在后一个周期输出，如下图所示

clip_image002[7]

    上述分析可以，一个4×4的乘加阵列可以计算一组4×4的乘加阵列完成计算，而对于其他维度的乘法，则可以通过多次调用的方式完成计算。譬如（4×4）×（4×8），可以将（4×8）的乘法拆分乘两个4×4的矩阵乘；而对于（4×8）×（8×4），两个矩阵计算完成后还需要将其结果累加起来，这也是为何TPU在乘加阵列后需要添加Accumulators的原因。最终脉动阵列设计如下所示（以4×4为例）

clip_image002[11]

2. 脉动阵列的实现
    如第一节所述，可通过HLS构建一个脉动阵列并进行仿真。类似TPU中的设计，采用INT8作为计算阵列的输入数据类型，为防止计算过程中的溢出，中间累加结果采用INT32存储。由于INT32的表示范围远高于INT8，认为计算过程中不存在上溢的可能性，因此没有对溢出进行处理。脉动阵列的计算结果数据类型为INT32，会在后文进行下一步处理。

    脉动阵列实现的关键代码包括

1. Feature向右侧移动

复制代码
for(int j=0;j<MXU_ROWNUM;j++){
    for(int k=MXU_ROWNUM+MXU_COLNUM-2;k>=0;k--){
        if(k>0)
            featreg[j][k] = featreg[j][k-1];
        else
            if(i<mxuparam.ubuf_raddr_num)
                featreg[j][k] = ubuf[ubuf_raddr][j];
            else
                featreg[j][k] = 0;
    }
}
复制代码
2. 乘法计算以及向下方移动的psum

复制代码
for(int j=MXU_ROWNUM-1;j>=0;j--){
    for(int k=0;k<MXU_COLNUM;k++){
        ap_int<32> biasreg;
        biasreg(31,24)=weightreg[MXU_ROWNUM+0][k];
        biasreg(23,16)=weightreg[MXU_ROWNUM+1][k];
        biasreg(15, 8)=weightreg[MXU_ROWNUM+2][k];
        biasreg( 7, 0)=weightreg[MXU_ROWNUM+3][k];
        if(j==0)
            psumreg[j][k] = featreg[j][k+j]*weightreg[j][k] + biasreg;
        else
            psumreg[j][k] = featreg[j][k+j]*weightreg[j][k] + psumreg[j-1][k];
    }
}
复制代码
    完成代码编写后可进行行为级仿真，可以看出整个计算阵列的时延关系

1. 对于同一列而言，下一行的输入比上一行晚一个周期

Screenshot from 2019-06-11 01-05-31

2. 对于同一行而言，下一列的输入比上一列晚一个周期（注意同一行输入数据是一样的）

Screenshot from 2019-06-11 01-06-08

3. 下一列的输出结果比上一列晚一个周期

Screenshot from 2019-06-11 01-07-39

 

3. 从矩阵乘法到三维卷积
    卷积神经网络计算过程中，利用kh×kw×C的卷积核和H×W×C的featuremap进行乘加计算。以3×3卷积为例，如下图所示，省略Channel方向，拆分kh和kw方向分别和featuremap进行卷积，可以得到9个输出结果，这9个输出结果按照一定规律加在一起，就可以得到追后的卷积计算结果。下图给出了3×3卷积，padding=2时的计算示意图。按F1-F9给9个矩阵乘法结果编号，输出featuremap中点（2，1）——指第二行第一个点——是F1（1，1），F2（1，2），F3（1，3），F4（2，1），F5（2，2），F6（2，3），F7（3，1），F8（3，2），F9（3，3）的和。

 clip_image002[13]

    下面的MATLAB代码阐明了这种计算三维卷积的方式，9个结果错位相加的MATLAB代码如下所示

复制代码
output = out1;
output(2:end,2:end,:) = output(2:end,2:end,:) + out2(1:end-1,1:end-1,:);
output(2:end,:,:) = output(2:end,:,:) + out3(1:end-1,:,:);
output(2:end,1:end-1,:) = output(2:end,1:end-1,:) + out4(1:end-1,2:end,:);
output(:,2:end,:) = output(:,2:end,:) + out5(:,1:end-1,:);
output(:,1:end-1,:) = output(:,1:end-1,:) + out6(:,2:end,:);
output(1:end-1,2:end,:) = output(1:end-1,2:end,:) + out7(2:end,1:end-1,:);
output(1:end-1,:,:) = output(1:end-1,:,:) + out8(2:end,:,:);
output(1:end-1,1:end-1,:) = output(1:end-1,1:end-1,:) + out9(2:end,2:end,:);
复制代码
    而在实际的HLS代码以及硬件实现上，部分未使用的值并未计算，因此实际计算的index和上述示意图并不相同，具体可参考testbench中的配置方法。

4. 其他
    GPU的volta架构中引入了Tensor Core来计算4×4的矩阵乘法，由于4×4的阵列规模较小，其内部可能并没有寄存器，设计可能类似第一节图1所示。由于其平均一个周期就能完成4×4矩阵计算，猜测采用第一节中阵列进行堆叠，如下图所示。

image

    一些FPGA加速库中利用脉动阵列实现了矩阵乘法，不过不同与TPU中将一个输入固定在MAC内部，还可以选择将psum固定在MAC内部，而两个输入都是时刻在变化的。这几种方式是类似的，就不再展开描述了。