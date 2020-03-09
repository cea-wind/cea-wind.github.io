title: 矩阵乘法加速器的设计框架
tags:
  - AI CHip
  - Architecture
  - Matrix Accelerator
categories:
  - AI Chip
mathjax: true
author: sea wind
date: 2020-03-09 21:12:52
---
>以往我分析了一些AI加速器的设计，包括TPU，FSD，华为达芬奇等，无一例外都是从已经给出的设计出发，去分析其优缺点和应用范围。在之前的文章中，关于这些设计是如何完成的，其背后是否有一定设计原则和理念的内容均没有进行探讨。而这两点，实则是设计一个优秀的，可持续迭代的加速器的基础。本文将从矩阵加速器出发，通过一些简化的模型，给出简单的设计框架。

# 1. 矩阵乘法和硬件模型

一般来说，矩阵乘法加速器中需要加速的计算可表示为

$$ C = A\times B + C $$

其中$A\in R^{m\times k}$, $B\in R^{k\times n}$, $C\in R^{m\times n}$ 。

矩阵乘法加速器，一般至少包括计算单元，缓存（SRAM等构成）和内存（譬如DDR等）。其中缓存的读写速率较高，可以和计算单元的运算速度相匹配，但容量较小；内存的容量相对缓存较大，但读写速率较低。

![](https://raw.githubusercontent.com/cea-wind/blogs_pictures/master/img20200309214550.png)


# 2. 带宽优化的矩阵乘法加速器设计

和一般的处理器相比，特定的加速器可以设计数量巨大的计算单元（譬如Google TPU V1设计了65536个乘法器）；但是DDR的带宽的提升却是有限的。因此，设计目标之一在于优化数据访问，降低DDR的读写带宽。

假设加速器的总缓存大小为$M$, 在一次计算过程中，用于存储矩阵$A,B,C$的缓存空间大小分别为$M_A,M_B,M_C$。

矩阵乘法加速器的设计目的一般是为了加速大规模的矩阵乘法计算，为了简化分析过程，假设矩阵$A,B,C$的大小$S_A,S_B,S_C$均远大于$M$，即计算过程中每次只能在缓存中存放一部分数据，完成子矩阵$A_{sub},B_{sub},C_{sub}$的计算。显然，存放在缓存中的数据都会参与运算，否在有冗余数据浪费存储和带宽。因此$A_{sub},B_{sub},C_{sub}$应能够完成一组矩阵计算，即

$$A_{sub}\in R^{p\times s},B_{sub}\in R^{s\times q},C_{sub}\in R^{p\times q}$$

据此，为了完成矩阵计算，从DDR到SRAM的总数据读写为

$$D_{size} = n/q \times S_A + m/p \times S_B + 2\times S_C$$

据此可以给出优化目标为

$$
\mathbf{min} : mnk/q + mnk/p +2mn \\ 
\mathbf{sub.to }: p\times s + s\times q + p\times q \leqslant M\\ 
 p>0,s>0,q>0
$$

简化为

$$
\mathbf{min} : 1/q + 1/p \\ 
\mathbf{sub.to }: p\times s + s\times q + p\times q \leqslant M\\ 
 p>0,s>0,q>0
$$

求解得当$s=1$，$p=q=\sqrt{M+1}-1$时得到最优解。即若要设计一个带宽优化的乘法器，应该尽可能的将缓存用于存储$C_{sub}$，每次计算的子矩阵为

$$C_{sub}^{p\times q} += A_{sub}^{p\times 1}  + B_{sub}^{1\times q} $$

Telsa的FSD的设计和上述讨论结果是一致的（只不过FSD的SRAM对应了上述的DDR，Register对应了上述的SRAM），FSD计算过程中$A_{sub}\in R^{96\times 1},B_{sub}\in R^{96\times 96},C_{sub}\in R^{96\times 96}$。对应的FSD的设计实际上是以降低SRAM-Register之间的读写为目的进行优化的。

# 3. 计算优化的矩阵乘法加速器设计
依据第二节的结果，每次计算的子矩阵为

$$C_{sub}^{p\times q} += A_{sub}^{p\times 1}  + B_{sub}^{1\times q} $$

整个计算过程中，其并行度最高为${p\times q}$（即每个周期完成${p\times q}$个乘法）。而为了完成一次计算，需要从缓存里读取$p+q+q\times q$个数据送入到计算阵列中。因此一次读/写的数据位宽宽度极高，随着并行度的增长，数据位宽线性增长。

数据位宽的问题主要存在$C_{sub}$上。为了解决这一问题，Telsa FSD采用了移位的方式，在计算完成后，将计算结果依次写回到SRAM中。

如果设计目的在于计算阵列和缓存之间的优化，参考第二节的设计思路，在一定并行度上，希望尽可能降低缓存的读写带宽，优化目标可以表示为

$$
\mathbf{min}:x\times y+y\times z+x\times z \\
\mathbf{sub.to }:x\times y\times z=P \\ 
 x>0,y>0,z>0
$$

其中$P$代表计算阵列的并行度，求解得当$x=y=z=\sqrt[3]{P}$时，此时设计的计算阵列对缓存的访问可以尽可能的低。

华为的达芬奇架构中计算阵列的设计和上述讨论是一致的，达芬奇中的CUBE Core是一个$16\times16\times16$的MAC阵列（以Davinci Max为例），可以完成
$$C_{sub}^{16\times 16} += A_{sub}^{16\times 16}  + B_{sub}^{16\times 16} $$
的矩阵计算。

# 4. 总结

上述的所有讨论都基于一个最简单的硬件模型，从两个角度分别求解了理论上最优的设计应该是怎么样的。

实际情况往往会复杂很多，硬件架构方面就会复杂很多。同时优化的目标往往有多个，而优化的限制条件也会有很多。

但是在我看来，只有采用这样的设计方法，即将问题建模，求解，才能造就一个好的设计。也只有采用这样的设计方法，才能再已有的基础上，进一步增加优化目标和优化条件，进一步的优化架构设计。

