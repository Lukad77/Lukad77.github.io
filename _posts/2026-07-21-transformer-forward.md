---
layout: post
title: Transformer 结构手推笔记（前向全流程）
date: 2026-07-21
category: llm-agent
tags: [深度学习, Transformer, 注意力机制, 生成式推荐]
description: 用一个 A→B→C、d=4、2 头的小例子，把整块 Transformer Block 的单次前向传播从头手算一遍：embedding + 位置编码 → Q/K/V 投影 → 缩放点积注意力 + 因果掩码 → 多头拼接 → 残差 + LayerNorm → FFN。每一步都有精确数值和代码骨架。
---

> 这一篇是 [从字符串到字符串](/2026/07/23/string-to-string/) 里 **②模型空间**的深挖：单次前向传播每一步矩阵运算怎么走。用一个小例子把整块 Transformer Block 从头手算一遍。序列 `A → B → C`，隐藏维度 `d=4`，故意取小方便手算。
>
> 姊妹篇：[从字符串到字符串](/2026/07/23/string-to-string/)（外层大局观 + tokenizer + 生成循环）与 *Transformer 训练与推理笔记*（本文里那些 W 矩阵怎么学出来的、反向传播、推理循环）。

> **配图约定**：本篇每一节各有一张图。**后一节的图会把前一节的成果画成灰虚线 + § 标签的子模块 box（内含关键缩略），本节新做的部分用紫色实线突出**——一路读下来，图会在你眼前一节一节长大，最后一张 §7-§9 图就是完整的 Transformer Block。

一句话主线：**embedding 查表 → 加位置编码 → Q/K/V 投影 → 缩放点积注意力（+因果掩码）→ 多头拼接 → 残差 + LayerNorm → FFN → 残差 + LayerNorm → Block 输出**，形状始终 `T×d`，故可堆 N 层。

---

## 0. 例子设定

**输入 embedding 矩阵 X（3×4）**：每个商品经 embedding 查表得到的向量。

```
        维度0 维度1 维度2 维度3
A     [   1    0    1    0  ]
B     [   0    1    1    0  ]
C     [   1    1    0    1  ]
```

**位置编码 P（3×4）**：为手算清爽，用可学习位置编码取干净值（正弦编码公式为 `PE(t,2i)=sin(t/10000^(2i/d))`，思想一致，数值更脏）。

```
pos0 [ 1 0 0 0 ]   pos1 [ 0 1 0 0 ]   pos2 [ 0 0 1 0 ]
```

**关键概念**

- **维度**：一个向量里有几个数。d=4 = 每个商品用 4 个数（4 维空间里的一个点）表示。单个维度通常无人类可读含义，含义体现在**向量间的几何关系**（点积大=相似）。
- 维度越多 → 刻画细粒度上限越高，但受数据/算力约束，过多会过拟合、涨成本，是 capacity vs cost 的权衡。

---

## 1. 位置编码相加 → X′

![§1 输入准备：ids → tok_emb → +pos_enc → x'](/assets/img/transformer-forward/sec1_input_prep.png)

`X′ = X + P`（逐元素相加）。

```
X′：
行0(A) [ 2 0 1 0 ]
行1(B) [ 0 2 1 0 ]
行2(C) [ 1 1 1 1 ]
```

**为什么需要位置编码**：自注意力本身对顺序不敏感（两两算相似度，与位置无关）。商品原始向量里没有"谁先谁后"的信息，顺序是靠**排列 + 位置编码 + 因果掩码**人为注入的。A/B/C 的先后 = 序列位置号 = 用户点击的时间顺序。

**位置编码的三代演进**（本例用干净数值方便手算，工程上主流是 RoPE）：

- **sin/cos 绝对编码**（原始 Attention Is All You Need）：`PE(t,2i)=sin(t/10000^(2i/d))`。确定性、可外推到没见过的长度，但只表达绝对位置。
- **RoPE 旋转位置编码**（LLaMA/Qwen 主流）：把位置信息以**旋转角度**注入 Q/K 内积，天然表达**相对位置**（两个位置的点积只取决于它们的距离，不取决于绝对位置）。长度外推更好。
- **ALiBi**（Attention with Linear Bias）：更简单粗暴，直接在注意力分数上按距离加一个线性偏置，越远惩罚越大。免训练即可外推到更长序列。

不管哪一代，思想都一致：**位置信息在注意力发生前就掺进 Q/K**，让"同一内容放不同位置"能算出不同的注意力。

**代码骨架**（sin/cos 版）：

```python
def sinusoidal_positional_encoding(max_len, d):
    pos = np.arange(max_len)[:, None]                        # (max_len, 1)
    i   = np.arange(d)[None, :]                              # (1, d)
    angle_rates = 1.0 / np.power(10000, (2 * (i // 2)) / d)  # 每维一个频率
    pe = np.zeros((max_len, d))
    pe[:, 0::2] = np.sin(pos * angle_rates[:, 0::2])         # 偶数维: sin
    pe[:, 1::2] = np.cos(pos * angle_rates[:, 1::2])         # 奇数维: cos
    return pe

# 用法：x = tok_emb[ids] + pos_enc[:T]     # 位置信息在投影前就掺进 x
```

---

## 2. 线性变换 → Q / K / V

![§2 Q/K/V 投影：§1 输入准备封装为灰虚线子模块，本章新增 3 次独立线性投影](/assets/img/transformer-forward/sec2_qkv_projection.png)

用三个权重矩阵（4×2，把 4 维压成 2 维；这里取干净 0/1 值）：

```
W^Q: 列0=x0+x2, 列1=x1+x3
W^K: 列0=x0+x3, 列1=x1+x2
W^V: 列0=x0+x2, 列1=x1+x2
```

得到（头1）：

```
      Q(问什么)      K(能答什么)     V(内容)
A  [  3,  0 ]     [  2,  1 ]     [  3,  1 ]
B  [  1,  2 ]     [  0,  3 ]     [  2*,...] → 实际 [1,3]
C  [  2,  2 ]     [  2,  2 ]     [  2,  2 ]
```

（V 精确值：V_A=[3,1], V_B=[1,3], V_C=[2,2]）

**Q/K/V 是同一个向量的三副面孔**：Q=当前想查什么，K=每个位置能提供什么（索引），V=每个位置实际的内容（被聚合对象）。拆开 W^Q/W^K 让"相关性"可不对称、可在特定子空间衡量；W^V 把"用于匹配的表示"和"传递的内容"解耦。

**位置信息如何进入 Q**：因为 P 在投影前就加进了 X，`Q = (X+P)W^Q = X·W^Q + P·W^Q`，后一项是"位置印章"。同一内容放不同位置，Q 不同（实证：itemB 内容放 pos0 → Q=[2,1]，放 pos1 → Q=[1,2]）。

**代码骨架**：Q/K/V 就是三次独立的线性投影：

```python
Q = linear(x, Wq, bq)   # (T, d) — 想找什么
K = linear(x, Wk, bk)   # (T, d) — 是什么(索引)
V = linear(x, Wv, bv)   # (T, d) — 能提供什么
# 三个投影权重独立训练，所以 Q / K / V 是同一 x 的三副不同"面孔"
```

---

## 3. 注意力分数（缩放点积）

> **§3-§5 三节共享一张图**：因为它们是同一个"单头软检索"的三个连续步骤。图里 §1/§2 是已封装的灰虚线子模块（当作黑盒复用），§3/§4/§5 用紫色实线三个连续小 box 表示。

![§3-§5 单头注意力：三步走完成一次软检索](/assets/img/transformer-forward/sec3_5_single_head.png)

`S = Q·Kᵀ / √d_k`，`d_k=2`，`√2≈1.414`。原始点积 `Q·Kᵀ`：

```
          →Key: A     B     C
Query A:        6     0     6
Query B:        4     6     6
Query C:        6     6     8
```

除以 √2：

```
      A      B      C
A [ 4.24   0.00   4.24 ]
B [ 2.83   4.24   4.24 ]
C [ 4.24   4.24   5.66 ]
```

**为什么除 √d_k**：维度越大点积越容易变大，数值太大会让 softmax 过尖锐、梯度趋 0。缩放把数值拉回合理区间，不改变相对大小。

---

## 4. 因果掩码 + softmax

**因果掩码**（Decoder 自回归的灵魂）：位置 i 只能看 j ≤ i，未来位置设 −∞。行=谁在看，列=被看的是谁，禁掉右上三角（j>i）：

```
      A      B      C
A [ 4.24   -∞    -∞   ]   ← A 只看自己
B [ 2.83  4.24   -∞   ]   ← B 看 A、B
C [ 4.24  4.24  5.66  ]   ← C 看全部
```

`exp(-∞)=0`，故未来位置权重必为 0，信息看不到。用 −∞ 而非删除，是为保持方阵、可并行。

逐行 softmax → 注意力权重：

```
      A     B     C
A [ 1.00   0     0   ]
B [ 0.20  0.80   0   ]
C [ 0.16  0.16  0.67 ]
```

读法：位置 B 的输出 = 20%A + 80%B；位置 C = 16%A + 16%B + 67%C。

---

## 5. 加权聚合 Value → 单头输出 Z

`Z = A·V`（搬运的是 V，不是 Q/K）：

```
Z_A = 1.00·V_A                       = [3.00, 1.00]
Z_B = 0.20·V_A + 0.80·V_B            = [1.40, 2.60]
Z_C = 0.16·V_A + 0.16·V_B + 0.67·V_C = [2.00, 2.00]
```

**单头注意力一句话**：Q/K 决定看谁（权重），V 决定搬什么（内容），掩码决定能看哪些。

**代码骨架**：§3-§5 三步合起来就是这五行——软检索的完整数学：

```python
def single_head_attention(Q, K, V, causal=True):
    T, d_head = Q.shape
    scores = Q @ K.T / np.sqrt(d_head)               # § 3  缩放点积打分
    if causal:
        mask = np.triu(np.ones((T,T)), k=1).astype(bool)
        scores = np.where(mask, -1e9, scores)         # § 4  因果掩码：未来位置 → -inf
    weights = softmax(scores, axis=-1)                # § 4  逐行 softmax 归一
    return weights @ V, weights                       # § 5  加权聚合 Value
```

---

## 6. 多头注意力

![§6 多头注意力：§3-§5 单头 SDPA 封装为灰虚线子模块，本章新增切 h 头 + concat + W^O](/assets/img/transformer-forward/sec6_multihead_layered.png)

把第 2~5 步用**多套 W^Q/W^K/W^V 并行跑几遍**，每套是一个"头"，最后拼接 + 输出投影 W^O。d=4 切成 h=2 头，每头 d_k=2。

- 各头在自己的 2 维子空间独立算注意力，学不同的"相关方向"（同品类/最近行为/…）。
- 改头数基本不改参数量（d_k=d/h，总参数≈d×d），多头是把容量重新切分给不同视角。
- 硬约束 `h | d`；经验甜点 d_k≈64（故常见 h=8/12/16 配 d=512/768/1024）。
- 头1 vs 头2 对 B 行：头1 主看自己(0.8)，头2 对 A/B 各半(0.5) —— 视角不同。
- 拼接把两头输出并排 → W^O 搅拌融合 → 映射回 d=4。输出形状 T×d 不变。

![多头注意力机制与内部 Q/K/V 关系](/assets/img/transformer-forward/multi_head_attention.png)

多头输出（W^O 取恒等的简化下）：

```
A [ 3.00 1.00 1.00 0.00 ]
B [ 1.40 2.60 1.00 0.00 ]
C [ 2.00 2.00 1.00 0.67 ]
```

**代码骨架**：一次投影出完整 Q/K/V，再沿最后一维切成 h 份，每份独立走一次 `single_head_attention`，最后拼回来过 W^O：

```python
def multi_head_attention(x, p):
    Q = linear(x, p["Wq"], p["bq"])                       # 三个大投影,一次算全
    K = linear(x, p["Wk"], p["bk"])
    V = linear(x, p["Wv"], p["bv"])
    dh = d // h                                            # 每头维度 d_k
    heads = []
    for i in range(h):                                     # 每头一份 (T, dh)
        Qi, Ki, Vi = Q[:, i*dh:(i+1)*dh], K[:, i*dh:(i+1)*dh], V[:, i*dh:(i+1)*dh]
        out_i, _ = single_head_attention(Qi, Ki, Vi, causal=True)
        heads.append(out_i)
    return linear(np.concatenate(heads, -1), p["Wo"], p["bo"])  # 拼回 (T,d) → W^O 融合
```

**多头在推理时的三个工程演进**（因为标准 MHA 每头都要缓存自己的 K/V，长上下文时显存/带宽是瓶颈）：

- **MQA（Multi-Query Attention）**：所有头**共享一套 K/V**，Q 仍是 h 套。KV 缓存缩小 h 倍，推理显存和带宽大幅降低；代价是牺牲一点质量。
- **GQA（Grouped-Query Attention）**：折中方案，把 h 个头分成 g 组，每组共享一套 K/V（LLaMA-2/3 用的就是这个）。质量和 MQA 的显存优势之间取平衡。
- **Flash Attention**：不改变数学，只重排计算顺序——**利用 GPU SRAM 分块处理、不落地 T×T 大矩阵**，从 IO 角度提速 2-4 倍且省显存。工业在线场景毫秒级推理很依赖它。

这三者都是"标准 MHA 太贵"催生的工程优化，数学上仍是同一套注意力，只是"缓存哪些 K/V"和"怎么算 softmax"变了。

---

## 7~9. 残差 + LayerNorm + FFN → 收口整块 Block

![§7-§9 Transformer Block(pre-norm)：§6 MHA 装进注意力子层，再叠 FFN 子层收口](/assets/img/transformer-forward/sec7_9_block.png)

骨架规律：**每个子层都包成 `LayerNorm( 输入 + 子层(输入) )`**。子层1=注意力，子层2=FFN。

**第 7 步：残差① + LayerNorm①**

残差 `res1 = X′ + MHA`：

```
A [ 5.00 1.00 2.00 0.00 ]
B [ 1.40 4.60 2.00 0.00 ]
C [ 3.00 3.00 2.00 1.67 ]
```

残差意义：给梯度修高速路（深层可训），且子层无用时输出≈输入，信息不丢——多堆一层最差也不更糟。

LayerNorm①（对**每一行**沿特征维归一化：减均值、除标准差，再乘 γ 加 β，此处 γ=1/β=0）。A 行 `[5,1,2,0]`：μ=2，σ²=3.5，σ=1.87 → `[1.60, -0.53, 0.00, -1.07]`。

```
LN① 输出 U：
A [ 1.60 -0.53  0.00 -1.07 ]
B [-0.36  1.56  0.00 -1.20 ]
C [ 0.98  0.98 -0.70 -1.26 ]
```

LayerNorm 意义：残差累加会让数值漂/爆，LN 把每个位置向量拉回均值0/方差1，稳住分布、训练更稳。（沿特征维、逐行独立，区别于跨样本的 BatchNorm。）

**第 8 步：FFN（逐位置）**

FFN = 两层 MLP：`W₂·act(W₁x+b₁)+b₂`，先扩张(d→4d)再收缩，ReLU 提供非线性。逐位置独立、共享权重。以 A 行为例（d_ff=4）：

```
U_A=[1.60,-0.53,0,-1.07]
扩张探测 z=[1.60,-0.53,1.07,-1.07]
ReLU门控 a=[1.60, 0.00,1.07, 0.00]  ← 两个负的被关掉
收缩     FFN(U_A)=[1.60,2.67,1.07,0.00]
```

**第 9 步：残差② + LayerNorm② → Block 输出**

`res2 = U + FFN(U)`：A 行 = `[3.20, 2.14, 1.07, -1.07]`；LayerNorm② 后 A 行 ≈ `[1.18, 0.51, -0.17, -1.52]`。

![Transformer Block 完整数据流](/assets/img/transformer-forward/transformer_block_flow.png)

**闭环**：进 3×4、出 3×4 → 可原样堆 N 层；末层取目标位置向量过"输出头(d→词表)+softmax"吐一个字。

**代码骨架**：LayerNorm + FFN + pre-norm block 三件套：

```python
def layer_norm(x, gamma, beta, eps=1e-5):
    mu  = x.mean(-1, keepdims=True)                       # 逐行(每个 token)独立
    var = x.var (-1, keepdims=True)
    return gamma * (x - mu) / np.sqrt(var + eps) + beta   # 拉回均值0方差1, 再缩放平移

def feed_forward(x, p):
    h = gelu(linear(x, p["W1"], p["b1"]))                 # (T, d)  → (T, d_ff)  升维 + 非线性
    return   linear(h, p["W2"], p["b2"])                  # (T, d_ff) → (T, d)  降回

def transformer_block(x, p):                               # pre-norm 版 (GPT/LLaMA)
    a, _ = multi_head_attention(layer_norm(x, p["ln1_g"], p["ln1_b"]), p)
    x = x + a                                              # 子层1 + 残差
    f = feed_forward         (layer_norm(x, p["ln2_g"], p["ln2_b"]), p)
    x = x + f                                              # 子层2 + 残差
    return x
```

> 变体：本例用原始论文 **Post-LN**（`LN(x+子层(x))`）；GPT/LLaMA 多用 **Pre-LN**（`x+子层(LN(x))`），训练更稳、更好堆深，思想相同仅 LN 位置不同。

---

## 附：两种架构范式

一句话：**去掉因果掩码就是 Encoder（双向自注意力，做"理解"）；拼上一条平行 encoder 输出 + 一次 cross-attention 就是 Encoder-Decoder（做"输入输出异构"任务）**——三种架构的算法内核都是这里手推的这套注意力。

我们本篇手推的（带因果掩码、单塔、无交叉注意力）就是 **Decoder-Only** 的自注意力，也是 GPT/LLaMA/Qwen 的做法。**Decoder-only 为什么会成为主流**（目标统一、数据无上限、好 scaling、蹭生态、以及代价）——这个大局观已经写在 [从字符串到字符串](/2026/07/23/string-to-string/) 的"为什么现在的大模型几乎都是 Decoder-only?"节里，此处不重复。

---

## 附：自注意力 vs 交叉注意力

**唯一区别 = Q/K/V 来自几条序列；点积→缩放→softmax→加权V 的算法完全相同。**

- **自注意力**：Q、K、V 全来自**同一条序列**（自己查自己）。
- **交叉注意力**：Q 来自 Decoder（目标），K/V 来自 Encoder 输出（源）——拿目标查源。
- 自注意力 = 交叉注意力在"两条序列合一"时的**特例**。
- 输入/输出（黑盒视角）：自注意力 1 条进→1 条出（T×d）；交叉注意力 2 条进（查询+源）→1 条出，**输出行数 = 查询序列行数**（不是源序列行数）。

![自注意力 vs 交叉注意力](/assets/img/transformer-forward/self_vs_cross_attention.png)

---

## 附：FFN 的本质

- 结构：两层 MLP（扩张→ReLU→收缩），逐位置独立、共享权重。
- **≡ 1×1 卷积**：跨位置共享权重（继承 CNN），但感受野=1（只看自己），因为跨位置混合归注意力管。原论文称其为"两个 kernel size=1 的卷积"。
- **key-value 联想记忆视角**：W₁ 每行=key（要匹配的模式），ReLU=匹配门，W₂ 对应列=value（要写入的内容）。输入匹配到哪些 key，就取回对应 value 加权叠加。**模型大量事实知识存在 FFN 参数里**。
- 与注意力的 KV 区别：注意力 KV 从当前序列**动态**算、跨**位置**取信息；FFN 的 KV **固定**在参数里、从**记忆库**取知识。

---

## 关键结论速查

1. embedding = 查表；维度=向量长度；含义在向量间几何关系。
2. 位置编码在投影前加入，把"位置印章"混进 Q/K/V，让注意力对顺序敏感。
3. Q/K 定权重、V 定内容、掩码定可见范围。
4. 缩放 √d_k 为数值稳定；因果掩码保证只看过去、且训练可并行。
5. 多头 = 并行多视角，不加参数只切分容量；输出回 T×d 靠 W^O。
6. 残差=梯度高速路+信息不丢；LayerNorm=逐行标准化稳分布；两者让 Block 可堆深。
7. FFN=逐位置非线性加工（1×1 卷积 / kv 记忆）；真正"输出字"的是末端输出头+softmax，不是 FFN。
8. 自注意力/交叉注意力算法同源，只差 Q/K/V 的序列来源。
