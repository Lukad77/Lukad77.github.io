---
layout: post
title: 从字符串到字符串：大模型收到一句话后到底发生了什么
date: 2026-07-23
category: llm-agent
tags: [Transformer, LLM, tokenizer, BPE, 端到端]
description: 从最外层看大模型完整链路：字符串怎么被 tokenizer 切成整数、进 Transformer 前向 N 次、再由 tokenizer 拼回字符串。覆盖 BPE 训练/编码手写 demo、Decoder-only 架构选型动机、端到端总纲图解。
---

# 从字符串到字符串：大模型收到一句话后到底发生了什么

> 从最外层往里看：一段字符串怎么被 tokenizer 切成整数、进 Transformer 前向 N 次、再由 tokenizer 拼回字符串。姊妹篇：*Transformer 结构手推笔记*（单次前向的每一步数学，是本篇②模型空间的深挖）与 *Transformer 训练与推理笔记*（参数怎么被梯度雕出来的、推理循环的机制）。

## 一个大模型收到用户输入之后，到底发生了什么？

一句话——**用户空间是离散字符，模型空间是整数序列，两个世界之间靠 tokenizer 这座双向桥连起来**。

用户输 "hello world"，tokenizer.encode 把它切成一串 id [9, 1, 25, 17, 18, 11, 3, 1]，Transformer 全程只跟这串整数打交道；等模型吐够了新 id，tokenizer.decode 再把它们拼回字符串还给用户。整个过程能画成一张总纲图（见下方**图 C**），里面标了两个 callout 分别指向我们之前拆过的两张局部图。

![图 C：大模型端到端总纲](/assets/img/llm-end-to-end/end_to_end_llm_pipeline.png)

**图 C：大模型端到端总纲**。三个区块：①用户空间（离散字符）、②模型空间（整数）、③回喂 & 结束（回到用户空间）。虚线 callout 指向之前的**图 A（自回归循环 · KV Cache）**和**图 B（最后位置的 Q 沿层增长）**——它们分别是这条主脉络的"时间维度"和"深度维度"局部放大。

## 为什么大模型走的是 Transformer 这条路？

因为 RNN 有两个致命伤——**慢**和**准**：

- **慢**：RNN 必须一个词一个词按时序算 `h_t = f(h_{t-1}, x_t)`，位置之间强依赖，无法并行。长序列训练奇慢。
- **准**：长距离依赖靠隐状态一路传递，路径过长梯度容易消失，"记不住"很久以前的信息。

Transformer 用注意力**一步到位让任意两个位置直接交互**——距离多远都一个矩阵乘就完事，既能并行、又没有距离衰减。这是它取代 RNN 的根本动机。

顺带一句：**Transformer 天生对顺序不敏感**（注意力是两两算相似度，与位置无关）。所以顺序信息是靠"位置编码 + 因果掩码"人为注入的——这两件事的具体细节在 *Transformer 结构手推笔记* 里手推展开。

## 那"字符串 → 整数"这一步到底在干嘛？

模型为什么不能直接吃字符串？因为 Transformer 的所有运算——注意力、FFN、LayerNorm——本质是**在实数矩阵上做线性代数**。它需要输入是"可以求梯度的数值"。字符串是离散的、变长的、没有可微结构的东西，没办法直接扔进矩阵乘。

所以问题转化成：**如何把任意一段人类文字，可逆地映射成一串整数？**——tokenizer 就是干这事的。它做的其实是两件小事：

1. **切分**：按训练好的规则把字符串切成一串"token"（词、子词、字节都可能）
2. **查表**：每个 token 在一个固定的 vocab（词表）里都有唯一整数 id，替换即可

vocab 一旦训练好就固定了（比如 GPT-2 是 50257 行、LLaMA-2 是 32000 行），每一行是一个 token 字符串，行号就是这个 token 的 id。**id 的本质就是"这个 token 在 vocab 里的行号"**——没有别的深意。

## BPE 是怎么切分的？——用手写 demo 看清楚

切分规则有好几种——BPE、WordPiece、SentencePiece——这里拿 GPT 系用的 BPE（Byte-Pair Encoding）讲透。核心四步：

1. **初始 vocab = 所有字符**（比如 `a-z` 加标点，加词尾标记 `</w>`）
2. **统计相邻 token 对的出现频率**
3. **合并最高频那一对**，作为一个新 token 加进 vocab，同时记下这条合并规则
4. **重复**直到 vocab 达到设定大小

encode 时，把输入按同样的合并规则贪心合一遍就行；decode 时反向拼接。

**训练循环的核心就四行**（伪代码等价于真代码）：

```python
for step in range(n_merges):
    pairs = get_pair_freqs(splits, word_freqs)              # 统计相邻对
    best_pair = max(pairs.items(), key=lambda kv: kv[1])[0] # 挑最高频那对
    splits = apply_merge(best_pair, splits)                 # 语料里把这对合并
    merges.append(best_pair)                                # 记下这条规则
```

在一个小玩具语料上训 15 次合并，就能看到 vocab 是怎么从字符长出来的：

```
初始 vocab(16 个字符): [' ', '</w>', 'b', 'd', 'e', 'g', 'h', 'i', 'l', 'm', 'n', 'o', 'r', 's', 't', 'w']

  合并 #1: ('l', 'o')       (频次 7) -> 新 token 'lo'
  合并 #2: ('w', 'e')       (频次 5) -> 新 token 'we'
  合并 #3: ('r', '</w>')    (频次 5) -> 新 token 'r</w>'
  合并 #4: ('t', ' ')       (频次 5) -> 新 token 't '
  合并 #5: ('w', 'i')       (频次 4) -> 新 token 'wi'
  合并 #6: ('wi', 'd')      (频次 4) -> 新 token 'wid'
  ...
训练结束,vocab 从 16 增长到 31(合并了 15 次)
```

看合并 #6 有意思：它把上一步刚合出来的 `'wi'` 又和 `'d'` 合成了 `'wid'`。**合并是层层套娃的**——所以 BPE 学出来的子词是"频繁的字节片段"，没有语法/词性概念，纯粹是统计意义上的高频组合。

### "频次 7" 是怎么数出来的？

`合并 #1: ('l', 'o') 频次 7` 里那个 7 不是拍脑袋——它是 `get_pair_freqs` 在整个语料的字符流上按词频加权数出来的。先看这次训练用的玩具语料：

```python
corpus = [
    "low", "lower", "lowest",                        # 各 1 次
    "newer", "newest", "newer",                      # newer 出现 2 次
    "wider", "wider", "widest", "wide",              # wider 出现 2 次
    "hello", "hello world", "hello world hello",     # 各 1 次
    "let it go", "let me be", "let it be",
]
```

`get_pair_freqs` 的核心也就四行——遍历每个词、拿相邻二元组、按词频加权累加：

```python
for word, symbols in splits.items():
    f = word_freqs[word]
    for a, b in zip(symbols, symbols[1:]):   # 抓所有相邻二元组
        pairs[(a, b)] += f                    # 按词频加权累加
```

拿 `(l, o)` 举例逐词分解，看这 7 票从哪来：

| 词 | 字符切分 | 出现 (l,o) 次数 | 词频 | 贡献票数 |
|---|---|---|---|---|
| `low` | `l o w </w>` | 1 | 1 | 1 |
| `lower` | `l o w e r </w>` | 1 | 1 | 1 |
| `lowest` | `l o w e s t </w>` | 1 | 1 | 1 |
| `hello` | `h e l l o </w>` | 1（第二个 l 后跟 o） | 1 | 1 |
| `hello world` | `h e l l o ␣ w o r l d </w>` | 1 | 1 | 1 |
| `hello world hello` | `… l l o ␣ w o r l d ␣ h e l l o </w>` | **2** | 1 | 2 |

加起来 `1+1+1+1+1+2 = 7`——就是这个 7 票。注意 `hello world hello` 一条独占 2 票，**票数不是"有几个词包含它"，而是"考虑重复和词频后的总投票"**。真实语料里 `the` 这种高频词一次能贡献几百万票，合并优先级极高。

排在 `(l,o)` 后面的候选是 `('w','e')` `('e','r')` `('r','</w>')` `('t',' ')`，都是 5 票——`(l,o)` 以 7 领先，所以第 1 轮就中标。**BPE 是贪心的**：每轮只挑当前最高频那一对，合完之后 splits 里所有 `l,o` 相邻位置立刻被替换成整体 token `'lo'`；下一轮再扫时，`'lo'` 已经是一个原子字符，可以再和旁边组新对——这正是下面**套娃合并**的起点。

顺带一个容易忽略的细节：`('t', ' ')` 也排进了 top 候选。**空格在这套 tokenizer 里就是普通字符**，和其它字节完全平权，可以参与合并投票——这也是真实 GPT-2 BPE 把"前导空格"当 token 一部分的哲学基础。

encode 时就把训练时的合并规则按顺序应用一遍，然后查表变 id：

```python
def encode(text, merges, vocab_id):
    ids = []
    for word in text.split(" "):
        symbols = word_to_symbols(word)          # 先按字符切
        for a, b in merges:                      # 按训练顺序应用每条合并规则
            i = 0
            while i < len(symbols) - 1:
                if symbols[i] == a and symbols[i+1] == b:
                    symbols = symbols[:i] + [a+b] + symbols[i+2:]
                else:
                    i += 1
        ids.extend(vocab_id[s] for s in symbols) # token 查表 -> 整数 id
    return ids
```

跑出来是这样：

```
输入 : 'lowest wider'
  切分 : ['lo', 'we', 'st</w>', 'wide', 'r</w>']
  ids  : [13, 26, 22, 30, 19]
  decode 回去: 'lowest wider'   一致? True
```

`'lowest'` 被切成 `'lo' + 'we' + 'st</w>'`——语料里 `'low'` `'we'` `'st'` 出现频次都够高，但 `'lowest'` 整体没高到能被合成一个 token。这个"高频子词就合、低频组合就拆"的经济学，是 BPE 处理**未见过的词**（out-of-vocabulary）的关键——只要子词都在 vocab 里，任何新词都能切出来。

## 选 tokenizer 到底在选什么？

>之前一直觉得 tokenizer 是个"预处理小工具"，看了 BPE 训练过程才发现——**vocab 本质上是从语料里学出来的一份"高频子串字典"**。它决定了什么概念可以被模型"一眼认出"、什么概念要被拆成好几片来处理。

所以：中文用 GPT-2 的 BPE 通常一个字被切成 2~3 个 sub-token（因为英文语料训练出的 vocab 里没几个中文字节的合并规则），效率就差；LLaMA-2 的 SentencePiece 训练时喂了中文语料，中文效率就好很多。换句话说，**tokenizer 是训练语料在 vocab 层面的固化**，选 tokenizer = 选一份"这个模型天生擅长哪种字节模式"的先验。

## vocab 是死的还是活的？

**不变**。tokenizer 是在训练开始**之前**就单独训好、然后冻住的。之后模型训练、微调、部署，vocab 都不动。原因有二：

1. 词嵌入表 `tok_emb` 的形状是 `(V, d)`，一旦 V 变了，`tok_emb` 就得重新初始化——之前学的向量全废
2. `lm_head` 输出维度也是 V。V 变意味着分类头的输出层要重建

所以**换 tokenizer 就等于换模型**——vocab 变了，id 的含义就全乱套了，emb 表和 lm_head 都得从零训。这也是为什么很多"扩词表微调"研究很谨慎，得先做好新 id 的嵌入初始化。

## 特殊 token 是啥？为什么要？

`<|endoftext|>` `<BOS>` `<EOS>` `<PAD>` 这些是**vocab 里预留出来的"控制 token"**——占 id 但不对应任何自然字符。用途：

- **BOS / EOS**：告诉模型"这里是文本开头 / 结尾"，让模型能够在没有 prompt 时基于 BOS 生成，或在合适位置自然停止
- **PAD**：批量推理时把变长序列补齐到同一长度，attention mask 里会屏蔽它们
- **system / user / assistant**：Chat 模型的角色分隔符，让模型知道"接下来这段是用户说的 / 我要说的"
- **工具调用 / 结构化输出的分隔符**：现代 LLM 常见

它们通常排在 vocab 的最前或最后，id 值固定，方便代码判断。

## 一步一步串起来看：从字符串到字符串的完整链路

用图 C 的编号对着走一遍：

**① 用户空间**：字符串 → tokenizer.encode → id 序列 `[id₁..id_T]`。这一步做的就是本文前面讲的切分+查表。

**② 模型空间**（Transformer 前向，每生成一个 token 跑一次完整前向）：
- `tok_emb` 查表：每个 id 换成一个 d 维向量 → `x` 形状变成 `(T, d)`
- 加位置编码 → `x⁽⁰⁾`（这时候模型只知道"每个 token 是谁、在第几位"）
- 过 N 层 Block（每层 = MHA + FFN + 残差 + LN），信息在位置间横向交流、在每个位置纵向深加工。**图 B** 展示了在这 N 层之中"最后位置的 Q 是怎么一层比一层更懂上下文"
- 最后 `ln_f` + `lm_head` 投影到词表维度 → logits `(T, V)`
- 取最后一行做 softmax → 下一个 token 的概率分布
- 采样 / argmax → 新 id ★

**整个前向其实就这几行**（numpy 手写版，剥掉注释后骨架清晰）：

```python
def gpt_forward(ids, params):
    x = params["tok_emb"][ids]                # (T,d) 查嵌入表：id -> d 维向量
    x = x + params["pos_enc"][:len(ids)]      # 加位置编码
    for layer in params["blocks"]:            # ×N 层
        x, _ = transformer_block(x, layer)    # 每层：MHA + 残差 + FFN + 残差
    x = layer_norm(x, params["lnf_g"], params["lnf_b"])
    logits = linear(x, params["lm_head_W"], params["lm_head_b"])  # (T, V)
    return logits
```

其中每层 block 里，注意力的核心又只是五行——"软检索"就藏在这里：

```python
scores = Q @ K.T / np.sqrt(d_head)                    # ① Q 对全体 K 打分
scores = np.where(causal_mask, -1e9, scores)          # ② 因果掩码：只准往回看
weights = softmax(scores, axis=-1)                    # ③ 归一成注意力权重
out = weights @ V                                     # ④ 按权重加权求和 V
# ⑤ 多头拼接后再过一次输出投影 W_o
```

**③ 回喂 & 结束**：
- 未结束：新 id 追加到 id 序列末尾，回到 ②。这里就是**图 A** 讲的自回归循环——每步只需要为最新位置算一份 Q/K/V，历史 K/V 走 KV Cache
- 已结束（触发 EOS / 达到最大长度 / 用户中断）：整段累积的 id 序列 → tokenizer.decode → 输出字符串还给用户

**两条虚线 callout 是维度分工**：图 A 是"时间维度"（多次前向之间怎么循环），图 B 是"深度维度"（单次前向之内 Q 怎么长出来）。图 C 把它俩夹在中间，就是端到端的全景。

## 为什么现在的大模型几乎都是 Decoder-only？

Transformer 原始论文画的是 Encoder-Decoder 双塔，但今天 GPT / LLaMA / Qwen 全部只用了一半。先看清三种架构的区别：

| 架构 | 注意力 | 参数分布 | 典型代表 | 擅长 |
|---|---|---|---|---|
| **Encoder-only** | 双向自注意力 | 集中 encoder | BERT | "理解"任务（分类、抽取），预训练用 MLM 完形填空 |
| **Encoder-Decoder** | Encoder 双向 + Decoder 因果 + **交叉注意力** | 分散三处 | T5 / BART | 输入输出异构（翻译、摘要） |
| **Decoder-only** | 只有统一的因果自注意力 | 集中 decoder | GPT / LLaMA / Qwen | "生成"任务（预测下一个 token） |

Decoder-only 之所以最终胜出，有四层原因：

1. **目标极其统一**：一切 NLP 任务都能表述成"预测下一个 token"——问答、翻译、摘要、代码、chat 全部塞进同一个训练目标里。目标越统一，越好把不同任务的数据混在一起训。
2. **数据无上限**：纯自监督，海量无标注文本直接可用，不需要人工标签。
3. **好 scaling、工程简单**：参数集中在 decoder 单塔，架构规整、算子简单，算力利用率高（OneRec-V2 报 MFU 20%+ vs Encoder-Decoder 5–10%）。而且能直接复用 LLaMA/Qwen 的开源框架和优化。
4. **蹭生态**：一旦 Decoder-only 的开源基座（LLaMA、Qwen、DeepSeek）跑通，所有下游微调/推理框架都自然贴上来，形成正反馈。

代价也有：单向建模、长上下文压力（原生 O(T²) 注意力）——用 RoPE/ALiBi、Sliding Window、KV Cache、Flash Attention 等手段缓解。

>我们端到端图 C 里画的正是 Decoder-only：单塔、只有一种因果自注意力、id 序列一路穿到底。**去掉因果掩码就变成 Encoder；再拼上一条平行的 encoder 输出 + 一次 cross-attention 就变成 Encoder-Decoder**——三种架构的算法内核完全同源，只在"注意力用不用 mask、Q/K/V 来自几条序列"两件事上区分。

这部分数学细节在 *Transformer 结构手推笔记* 的"自注意力 vs 交叉注意力"章节展开。

## 生产环境的 tokenizer 跟这个 demo 差多远？

这个 15 次合并的玩具 BPE 跟真正的 GPT 主要差在四点：

1. **规模**：真实 vocab 是几万~十几万；训练语料是几十~几百 GB 网页文本；合并次数上万
2. **字节级 BPE**（GPT-2/4）：先把文本 UTF-8 编码成字节，在字节层面做 BPE。好处是**任何 Unicode 都能被表示**，坏处是中文一个字要占多个字节 token
3. **前导空格作为 token 的一部分**：`'hello'` 和 `' hello'` 是不同的 token id。这样"the "是一个 token，模型不用每次重新学习"词后要跟空格"
4. **规范化**：像 SentencePiece 会先把文本做 Unicode 规范化（NFKC）、把空格换成特殊字符 `▁`（防歧义），再做 BPE

除此之外，**核心思想跟我们手写的 demo 完全一样**：字符对合并、贪心 encode、反向 decode。这也是为什么读一读 tiktoken / SentencePiece 源码不会太费劲——它们只是把上面的 demo 做工业化了。

## 落点收束

三句话把主脉络钉住：

1. **tokenizer 是离散字符空间和整数序列空间之间的双向桥**——不是预处理小工具，而是决定"这个模型能高效表达哪些字节模式"的先验
2. **vocab 一旦训好就冻住**——`tok_emb`、`lm_head` 都建立在它之上，换 tokenizer 就等于换模型
3. **端到端链路只做一件事**：`str →(tokenizer.encode)→ ids →(Transformer 前向 × N 次)→ ids →(tokenizer.decode)→ str`。中间那 N 次前向就是我们之前拆过的自回归循环 + 深度堆叠
