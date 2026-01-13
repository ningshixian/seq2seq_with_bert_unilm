# seq2seq_with_bert_unilm

- 参考《[Seq2Seq+前缀树：检索任务新范式（以KgCLUE为例）](https://www.spaces.ac.cn/archives/8802)》
- 项目介绍：[https://www.yuque.com/ningshixian/xqbgmt/qhdftxgmh7gh00ft](https://www.yuque.com/ningshixian/xqbgmt/qhdftxgmh7gh00ft)

用UNILM方案来训练一个Seq2Seq模型。将用户上下文 query 当作Seq2Seq的输入，将坐席客服回复（type=out）用[SEP]连接起来作为目标；推理/解码的时候，我们先把所有的坐席回复（即答案）建立成前缀树，然后按照前缀树进行 [beam search 解码](https://www.yuque.com/ningshixian/pz10h0/occzbt) and 输出结果。

**UniLM 的 Mask 机制**

**借鉴了 **[UNILM](https://arxiv.org/abs/1905.03197) 方案，**通过添加一个特别的Mask矩阵**，直接用单个Bert模型实现 Seq2Seq LM 任务（无需修改模型架构，且可以直接沿用Bert的 Masked Language Model 预训练权重）。Mask矩阵如下图所示，**作用是让Bert输入部分的Attention是双向的，输出部分的Attention是单向，从而满足Seq2Seq的要求。**

**带前缀树约束的 Beam Search 逻辑**

**注意，利用前缀树约束Seq2Seq解码其实很简单，即根据树上的每一条路径（以[BOS]开头、[EOS]结尾）查找到以某个前缀开头的字/句有哪些。然后，把模型预测其他字的 Logits 设置为**![](https://cdn.nlark.com/yuque/__latex/69232d85e66f774b5c6cc24df7d1d847.svg)（或者概率置零），保证解码过程只走前缀树的分支，而且必须走到最后，这样可以大幅缩减搜索空间及计算开销，同时也保证了生成的句子必然在话术库中；


---



**[bert4keras](https://github.com/bojone/bert4keras/tree/master) 提供的参考示例：**

* [task_seq2seq_ape210k_math_word_problem.py](https://github.com/bojone/bert4keras/tree/master/examples/task_seq2seq_ape210k_math_word_problem.py): 任务例子，通过[UniLM](https://kexue.fm/archives/6933)式的Seq2Seq模型来做小学数学应用题（数学公式生成），详情请见[这里](https://kexue.fm/archives/7809)。
* [从语言模型到Seq2Seq：Transformer如戏，全靠Mask](https://www.spaces.ac.cn/archives/6933)
  * [task_seq2seq_autotitle.py](https://github.com/bojone/bert4keras/tree/master/examples/task_seq2seq_autotitle.py): 任务例子，通过[UniLM](https://kexue.fm/archives/6933)式的Seq2Seq模型来做新闻标题生成。
  * [task_seq2seq_autotitle_csl.py](https://github.com/bojone/bert4keras/tree/master/examples/task_seq2seq_autotitle_csl.py): 任务例子，通过[UniLM](https://kexue.fm/archives/6933)式的Seq2Seq模型来做论文标题生成，包含了评测代码。
  * [task_seq2seq_autotitle_csl_mt5.py](https://github.com/bojone/bert4keras/tree/master/examples/task_seq2seq_autotitle_csl_mt5.py): 任务例子，通过[多国语言版T5](https://kexue.fm/archives/7867)式的Seq2Seq模型来做论文标题生成，包含了评测代码。
  * [task_seq2seq_autotitle_multigpu.py](https://github.com/bojone/bert4keras/tree/master/examples/task_seq2seq_autotitle_multigpu.py): 任务例子，通过[UniLM](https://kexue.fm/archives/6933)式的Seq2Seq模型来做新闻标题生成，单机多卡版本。
* [task_reading_comprehension_by_seq2seq.py](https://github.com/bojone/bert4keras/tree/master/examples/task_reading_comprehension_by_seq2seq.py): 任务例子，通过[UniLM](https://kexue.fm/archives/6933)式的Seq2Seq模型来做[阅读理解问答](https://kexue.fm/archives/7115)，属于自回归文本生成。
* [task_question_answer_generation_by_seq2seq.py](https://github.com/bojone/bert4keras/tree/master/examples/task_question_answer_generation_by_seq2seq.py): 任务例子，通过[UniLM](https://kexue.fm/archives/6933)式的Seq2Seq模型来做[问答对自动构建](https://kexue.fm/archives/7630)，属于自回归文本生成。
