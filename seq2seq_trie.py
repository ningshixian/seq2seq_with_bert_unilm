# Seq2Seq + trie
#   用UNILM方案来训练一个Seq2Seq模型，通过前缀树约束解码；
#   介绍链接：https://kexue.fm/archives/8802
# 基础模型为 UER；
#   Large版本推荐用腾讯UER开源的权重，原本是PyTorch版的，笔者将它转换为TF版了
#   下载地址：https://pan.baidu.com/s/1Xp_ttsxwLMFDiTPqmRABhg（提取码l0k6）

"""命令
nohup python seq2seq_trie.py > logs/seq_log.txt 2>&1 &
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # "0, 1"

import pickle
import re  
# import json
import ujson as json
import numpy as np
import traceback
import tensorflow as tf
from bert4keras.backend import keras, K
from bert4keras.layers import Loss
from bert4keras.models import build_transformer_model
from bert4keras.tokenizers import Tokenizer, load_vocab
from bert4keras.optimizers import Adam
import codecs
from bert4keras.snippets import sequence_padding, open
from bert4keras.snippets import DataGenerator
from AutoRegressiveDecoder import AutoRegressiveDecoder
from bert4keras.snippets import longest_common_subsequence as lcs
from keras.models import Model
from tqdm import tqdm
import pylcs
# from script.evaluation import evaluate
from rouge import Rouge  # pip install rouge
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import pandas as pd 

# sets random seed
seed = 123
np.random.seed(seed)
# set GPU memory
config = tf.ConfigProto()
config.gpu_options.allow_growth = True  # 按需求增长
sess = tf.Session(config=config)
keras.backend.tensorflow_backend.set_session(sess)


def get_contextid_2_oa(csv_file="data/contextid_2_oa.csv"):
    """跑出所有seat_account
    """
    contextid_2_oa = {}
    temp = pd.read_csv(csv_file)
    for index,row in temp.iterrows():
        contextid_2_oa[row["context_id"]] = row["seat_account"]
    return contextid_2_oa


def load_data(data):
    """
    train_origin_data是List[pandas.DataFrame]格式
    test_data是List[Dict]格式
    """
    D = []  # 存储<q,a>
    M = []  # 存储<a_richtext>
    contextid_2_oa = get_contextid_2_oa()   # context_id → oa_account
    oa_2_id = {}  # oa_account → oa_id
    for df in data:
        # 开始组装 Q-A 对训练数据
        pre_msg = [""] * first_n_sentences
        pre_ans = []
        pre_ans_m = []
        for index, row in df.iterrows():
            m, t = str(row["message"]).lower(), str(row["text"]).lower()  # 原始句子 处理后句子
            if row["calltype"] == "in":
                if (
                    pre_ans
                    and not set(pre_msg[-first_n_sentences:]).issubset(user_dict)
                ):
                    context = pre_msg[-first_n_sentences:]
                    answer = "。".join(pre_ans)  # 用句号将连续回复拼接起来
                    D.append((context, answer))
                    pre_ans = []
                    M.append((oa_id, " ".join(pre_ans_m)))
                    pre_ans_m = []
                # 过滤上下文中的重复句
                if not t==pre_msg[-1]:
                    pre_msg.append(t)
            elif row["calltype"] == "out":
                if row["contextid"] in contextid_2_oa:
                    oa_account = contextid_2_oa[str(row["contextid"])]
                    oa_2_id.setdefault(oa_account, len(oa_2_id)+1)
                    oa_id = oa_2_id[oa_account]
                else:   # 465
                    # oa_account = "None"
                    oa_id = 0
                # 过滤重复答案
                if (pre_ans and t == pre_ans[-1]) or (pre_ans_m and m==pre_ans_m[-1]):
                    continue
                # if "反馈一下" in m or "反馈一下" in t:
                #     continue
                while t and t[-1] in ["。", ".", '!', ',', '?', '！', '？', '，', '~',';','；']:
                    t = t[:-1]
                pre_ans.append(t)   # t可能为空，没关系？
                pre_ans_m.append(m)
        
        # 召回可能遗漏的 Q-A 对
        if (
            pre_ans
            and not set(pre_msg[-first_n_sentences:]).issubset(user_dict)
        ):
            context = pre_msg[-first_n_sentences:]
            answer = "。".join(pre_ans)  # 用句号将连续回复拼接起来
            D.append((context, answer))
            M.append((oa_id, " ".join(pre_ans_m)))  # 用空格将连续富文本回复拼接起来

    # 保存 oa_account → oa_id 的映射字典，用于推理阶段
    with codecs.open("data/oa_account_2_id.pkl", "wb") as tf:
        pickle.dump(oa_2_id,tf)

    # print(oa_2_id)
    # # {'qd_zhangjialin': 1, 'w_yanqing8': 2, 'qd_tongjie': 3, 'ranran-qd': 4, nan: 5, 
    # # 'qd_tangchuan': 6, 'qd_wangyilin': 7, 'qd_wangxianqin01': 8, 'qd_sunyiming': 9, 
    # # 'w_zhangli408': 10, 'w_zhouyongfang': 11, 'qd_yuansihan': 12, 'w_yangxintong1': 13, 
    # # 'w_taoyuqin3': 14, 'wanghaoyu': 15, 'w_xiangliangshuai': 16, 'qd_zhaoyan': 17, 
    # # 'qd_liaowentao': 18, 'qd_litao': 19, 'a-yuxiaotong': 20, 'qd_wsl': 21, 'lihuiwu': 22, 
    # # 'undefined': 23, 'w_zhangrongfei2': 24}
    return D, M


def load_test_data(data):
    """
    train_origin_data是List[pandas.DataFrame]格式
    test_data是List[Dict]格式
    """
    D, M = [], []
    for d in data:
        # d={q:xx, a:yy, label:0/1, a_html:yy}
        q,a,l = str(d["q"]).lower(), str(d["a"]).lower(), d["label"]
        a_html = str(d["a_html"]).lower() if "a_html" in d else ""
        if l == 0:  # 负例不带 a_html 字段！
            pass
        elif l == 1:
            pre_msg = [""] * first_n_sentences
            context = (pre_msg + q.split("[sep]"))[-first_n_sentences:]
            D.append((context, a))
            if a_html:
                M.append(a_html)
    return D, M


class Trie(object):
    """自定义Trie树对象，用来保存知识库
    """

    def __init__(self, value_key=-1):
        self.data = {}
        self.value_key = str(value_key)

    def __setitem__(self, key, value):
        """传入一对(key, value)到前缀树中
        """
        data = self.data
        for k in key:
            k = str(k)
            if k not in data:
                data[k] = {}
            data = data[k]
        # if self.value_key in data:
        #     if value not in data[self.value_key]:
        #         data[self.value_key] += "\t" + value
        # else:
        #     data[self.value_key] = value
        data[self.value_key] = value

    def __getitem__(self, key):
        """获取key对应的value
        """
        data = self.data
        for k in key:
            k = str(k)
            data = data[k]
        return data[self.value_key]

    def next_ones(self, prefix):
        """获取prefix后一位的容许集
        """
        data = self.data
        for k in prefix:
            k = str(k)
            data = data[k]
        return [k for k in data if k != self.value_key]

    def keys(self, prefix=None, data=None):
        """获取以prefix开头的所有key
        """
        data = data or self.data
        prefix = prefix or []
        for k in prefix:
            k = str(k)
            if k not in data:
                return []
            data = data[k]
        results = []
        for k in data:
            if k == self.value_key:
                results.append([])
            else:
                results.extend([[k] + j for j in self.keys(None, data[k])])
        return [prefix + i for i in results]

    def save(self, filename):
        with open(filename, "w") as f:
            json.dump(self.data, f, ensure_ascii=False)

    def load(self, filename):
        with open(filename) as f:
            self.data = json.load(f)


# # 测试前缀树
# KG = Trie()
# # key = [872, 1962, 102, 872, 738, 1962]
# KG[[872, 1962]] = "首都"
# KG[[872, 1962, 78]] = "capital"
# KG[[872, 1962, 44]] = "capital2"
# KG[[23, 82, 62]] = "冲冲冲"
# # KG.save('data/KG.json')
# print(KG[[872, 1962]])  # get 首都
# print(KG.next_ones([872, 1962]))    # ['78', '44']
# print(KG.keys([872]))   # [[872, '1962'], [872, '1962', '78'], [872, '1962', '44']]
# KG = Trie()
# KG.load("data/KG.json")
# print(KG.keys([872])) 
# exit()


def pre_tokenize(text):
    """单独识别出[xxx]的片段, 预分词
    """
    tokens, start = [], 0
    for r in re.finditer("\[[^\[]+\]", text):
        tokens.append(text[start : r.start()])
        tokens.append(text[r.start() : r.end()])
        start = r.end()
    if text[start:]:
        tokens.append(text[start:])
    return tokens


# 基本参数
max_q_len = 32
max_a_len = 64
batch_size = 16

# 模型路径
# pm_root = "../corpus/chinese_wwwm_ext_L-12_H-768_A-12"
# pm_root = "../corpus/chinese_roformer-sim-char-ft_L-12_H-768_A-12"    # RoFormer+UniLM+对比学习+BART+蒸馏
pm_root = "../corpus/mixed_corpus_bert_base_model"  # UER bert
config_path = pm_root + "/bert_config.json"
checkpoint_path = pm_root + "/bert_model.ckpt"
dict_path = pm_root + "/vocab.txt"

# user_dict = ["[pic]", "[know]", "[http]", "[alnum]", "[phone]"]  # 针对 e15
# user_dict = ["[pic]", "[know]", "[http]", "[alnum]", "[phone]", "[subphone]"]  # 针对 e20
user_dict = ["[pic]", "[know]", "[http]", "[alnum]", "[phone]", "[subpn]", "[ques]", "[time]", "[json]"]    # 自定义特殊占位符

# 建立分词器，添加特殊占位符special token (只能是小写)
# # 如何正确地在vocab里增加一些token？ #408
# # 怎么把special token人为的插入？ #403
# token_dict, keep_tokens = load_vocab(dict_path=dict_path, simplified=True)  # 精简词表,过滤冗余部分token,会导致解码异常！
token_dict = load_vocab(dict_path=dict_path)
pure_tokenizer = Tokenizer(token_dict.copy(), do_lower_case=True)  # 全小写
for special_token in user_dict:
    if special_token not in token_dict:
        token_dict[special_token] = len(token_dict)
compound_tokens = [pure_tokenizer.encode(s)[0][1:-1] for s in user_dict]
tokenizer = Tokenizer(token_dict, do_lower_case=True, pre_tokenize=pre_tokenize)


# # 测试分词器
# print(tokenizer.encode('你好', '你也好'))
# print(tokenizer.tokenize("句子一[pic]句子二"))
# print(tokenizer.encode("句子一[know]句子二"))
# print(tokenizer.tokenize("北京是[unused1]中国的首都"))
# q1, q2 = "句子一[SEP]句子二".split("[SEP]")[-2:]
# a_ids = tokenizer.encode(q1)[0]
# b_ids = tokenizer.encode(q2)[0][1:]
# token_ids = a_ids + b_ids
# segment_ids = [0] * len(a_ids)
# segment_ids += [1] * len(b_ids)
# print((token_ids, segment_ids))
# exit()


class data_generator(DataGenerator):
    """数据生成器
    单条样本：[CLS] Q1 [SEP] Q2 [SEP] Q3 [SEP] A [SEP]
    """

    def __iter__(self, random=False):
        batch_token_ids, batch_segment_ids = [], []
        for is_end, (q, a) in self.sample(random):
            q_ids = tokenizer.encode(q[0], q[1], maxlen=2 * max_q_len)[0]
            q_ids += tokenizer.encode(q[2], maxlen=max_q_len + 1)[0][1:]
            a_ids = tokenizer.encode(a, maxlen=max_a_len + 1)[
                0
            ]  # , maxlen=maxlen // 2 + 1
            token_ids = q_ids + a_ids[1:]  # [:maxlen]
            segment_ids = [0] * len(q_ids)
            segment_ids += [1] * (len(token_ids) - len(q_ids))

            batch_token_ids.append(token_ids)
            batch_segment_ids.append(segment_ids)
            if len(batch_token_ids) == self.batch_size or is_end:
                batch_token_ids = sequence_padding(batch_token_ids)
                batch_segment_ids = sequence_padding(batch_segment_ids)
                yield [batch_token_ids, batch_segment_ids], None
                batch_token_ids, batch_segment_ids = [], []


class CrossEntropy(Loss):
    """交叉熵作为loss，并mask掉输入部分
    """

    def compute_loss(self, inputs, mask=None):
        y_true, y_mask, y_pred = inputs
        y_true = y_true[:, 1:]  # 目标token_ids
        y_mask = y_mask[:, 1:]  # segment_ids，刚好指示了要预测的部分
        y_pred = y_pred[:, :-1]  # 预测序列，错开一位
        loss = K.sparse_categorical_crossentropy(y_true, y_pred)
        loss = K.sum(loss * y_mask) / K.sum(y_mask)  # 通过 Mask 使得只计算目标部分的损失
        return loss


# 模型构造
#   注意build_transformer_model中只要设置application='unilm'，
#   就会采用UNILM的思路自动加载Bert的MLM预训练权重，并且传入对应的Mask
model = build_transformer_model(
    config_path,
    checkpoint_path,
    model="bert",  # roformer
    application="unilm",
    # keep_tokens=keep_tokens,  # 只保留keep_tokens中的字，精简原字表
    compound_tokens=compound_tokens,  # 增加词，用字平均来初始化
)
# input("b")

# seq2seq 在生成输出序列的时候是一个time_step生成一个字，换句话说，在每个时间步都在解决一个分类问题。
# 所以这里选择最常用的交叉熵损失函数，但需要利用上述的Mask来屏蔽掉上下文部分的loss
output = CrossEntropy(2)(model.inputs + model.outputs)  # [token_ids, segment_ids, prediction_ids]

model = Model(model.inputs, output)
model.compile(optimizer=Adam(2e-5))  # lr
#model.summary()


class AutoQA(AutoRegressiveDecoder):
    """seq2seq解码器

    inputs模型的输入
    output_ids输出的id，就是模型当前解析的输出
    """

    @AutoRegressiveDecoder.wraps(default_rtype="probas")
    def predict(self, inputs, output_ids, states):  # 每一个时间步会执行一次
        token_ids, segment_ids = inputs
        # output_ids = np.empty((2,1), dtype=int)?
        all_token_ids = np.concatenate([token_ids, output_ids], 1)
        segment_ids = np.concatenate([segment_ids, np.ones_like(output_ids)], 1)
        probas = self.last_token(model).predict(
            [all_token_ids, segment_ids]
        )  # 创建一个只返回最后一个token的分布输出的新Model
        # 前缀树约束解码
        new_probas = np.zeros_like(probas)
        for i, ids in enumerate(output_ids):
            next_ids = [int(j) for j in KG.next_ones(ids)]  # 下一位容许集
            # # 加入自研的“前瞻”策略
            # if len(next_ids) > 1 and self.end_id in ids:  # 容许集大于1且已解码出S
            #     candidates = KG.keys(list(ids))  # 可能解码结果
            #     weights = np.ones_like(probas[i])  # 默认权重为1
            #     lcs0 = lcs(ids, token_ids[i])[0]  # 当前已经覆盖的token数
            #     for c in candidates:
            #         if len(c) > len(ids):
            #             c = [int(j) for j in c]
            #             w = lcs(c, token_ids[i])[0] - lcs0  # 未来还可能覆盖的token数
            #             weights[c[len(ids)]] = max(w + 1, weights[c[len(ids)]])
            #     probas[i] = np.power(probas[i], 1. / weights)  # 按 p^(1/n) 来增大权重
            if not next_ids:  # 如果容许集为空，意味着要结束了
                next_ids.append(self.end_id)
            new_probas[i, next_ids] += probas[i, next_ids]  # 只保留容许集概率
        new_probas /= new_probas.sum(axis=1, keepdims=True)  # 重新归一化
        return new_probas

    def generate(self, text, topk=1, stage="train"):
        q = text
        token_ids = tokenizer.encode(q[0], q[1], maxlen=2 * max_q_len)[0]
        token_ids += tokenizer.encode(q[2], maxlen=max_q_len + 1)[0][1:]
        segment_ids = [0] * len(token_ids)
        # 要输出多个结果时用random sample，要输出最大概率的单个结果时用beam search
        output_ids, output_scores = self.beam_search(
            [token_ids, segment_ids], topk=topk
        )
        # output_ids, output_scores = self.random_sample(
        #     [token_ids, segment_ids], n=topk, topk=topk   # 从概率最高的topk个中采样1个，组成n个结果
        # )
        if stage=="train":    # 仅用于训练阶段
            return [tokenizer.decode(x) for x in output_ids], output_scores
        elif stage=="infer":    # 仅用于推理阶段
            return [KG[x[:-1]] for x in output_ids], output_scores


autoqa = AutoQA(start_id=None, end_id=tokenizer._token_end_id, maxlen=max_a_len+1)
# 句子存入前缀树长度不设限，而解码beam search会限制最大长度，导致后面的 token 被直接丢弃，报错 KeyError: '-1'（对前缀树句子长度加限制！）
# print("tokenizer._token_end_id: ", tokenizer._token_end_id)     # 102


def just_show():
    examples = [
        ["", "", "你好，我想请问一下大屏应援活动是怎么搞的呢"],
        ["珑珠积分补录", "我支付宝付的，没法补录", "在盒马买的"],
        ["[PIC]", "[PIC]", "购物小票已经遗失。你们可以和商家索取一下"],  # '好的亲~为您记录反馈'
        ["", "人工客服", "你好，嗯，你们现在有那个推广活动是吗"],
        ["", "", "请问怎么通过小票录入的积分"],
        ["", "请问怎么通过小票录入的积分", "支付宝"],
        ["请问怎么通过小票录入的积分", "支付宝", "是的"],  # '[KNOW]'
        ["", "左庭右院", "消费了 怎么积分"],  # '[KNOW]'
        ["可以免费停车啊", "杭州", "滨江"],  # '车牌号提供一下呢'
        ['苏州新区龙湖天街', '因为疫情停业了', '什么时候恢复呀'],
        ['刷卡消费如何补录积分', '你好补录积分', '刷卡'],
    ]
    for s in examples:
        print(s)
        print(u"生成话术回复:", autoqa.generate(s, topk=3))
    print()


# class Evaluator(keras.callbacks.Callback):
#     """评估与保存
#     """

#     def __init__(self):
#         self.lowest = 1e10

#     def on_epoch_end(self, epoch, logs=None):
#         # 保存最优
#         if logs["loss"] <= self.lowest:
#             print("保存当前模型...")
#             self.lowest = logs["loss"]
#             model.save_weights("./best_model.weights")
        

class Evaluator(keras.callbacks.Callback):
    """评估与保存 rouge && bleu
    """
    def __init__(self):
        self.rouge = Rouge()
        self.smooth = SmoothingFunction().method1
        self.best_bleu = 0.

    def on_epoch_end(self, epoch, logs=None):
        if epoch>=10:
            metrics = test_predict(test_data, "data/predict.json", topk=3)  # 评测模型
            if metrics['bleu'] > self.best_bleu:
                print("保存当前模型...")
                self.best_bleu = metrics['bleu']
                model.save_weights('best_model.weights')  # 保存模型
            metrics['best_bleu'] = self.best_bleu
            # print('test_data:', metrics)


def test_predict(test_data, out_file, topk=1):
    """输出测试结果到文件
    """
    total = 0
    rouge_1, rouge_2, rouge_l, bleu = 0, 0, 0, 0
    rouge = Rouge()
    smooth = SmoothingFunction().method1

    fw = open(out_file, "w")
    for i in tqdm(range(len(test_data))):
        total += 1
        context, answer = test_data[i]

        try:
            pred_answer, proba = autoqa.generate(context, topk=topk)
        except Exception as e:
            print(traceback.format_exc())  # 错误日志 repr(e)

        l = {"q": context, "a": pred_answer, "gold": answer}
        l = json.dumps(l, ensure_ascii=False)
        fw.write(l + "\n")

        # Rouge && BLEU 指标评测
        answer = " ".join(answer).lower()
        pred_answer = " ".join(pred_answer).lower()
        if pred_answer.strip():
            scores = rouge.get_scores(hyps=pred_answer, refs=answer)
            rouge_1 += scores[0]['rouge-1']['f']
            rouge_2 += scores[0]['rouge-2']['f']
            rouge_l += scores[0]['rouge-l']['f']
            bleu += sentence_bleu(
                references=[answer.split(' ')],
                hypothesis=pred_answer.split(' '),
                smoothing_function=smooth
            )
    fw.close()

    rouge_1 /= total
    rouge_2 /= total
    rouge_l /= total
    bleu /= total
    print("Rouge && BLEU 指标评测结果如下：")
    print('rouge-1: {}, rouge-2: {}, rouge-l: {}, bleu: {}'.format(rouge_1, rouge_2, rouge_l, bleu))
    # rouge-1: 0.254695, rouge-2: 0.1127038, rouge-l: 0.221645, bleu: 0.05940
    return {
        'rouge-1': rouge_1,
        'rouge-2': rouge_2,
        'rouge-l': rouge_l,
        'bleu': bleu,
    }


KG_1 = Trie()   # 前缀树存储，用于训练
KG_2 = Trie()   # 以坐席id开头进行前缀树存储，用于推理
KG = Trie()

# # 测试坐席账号有哪些回复
# ans_list = []
# KG_2.load("data/KG_2.json")
# with codecs.open("data/oa_account_2_id.pkl", "rb") as tf:
#     oa_2_id = pickle.load(tf)
# seat_account = "qd_liaowentao"
# oa_id = oa_2_id.get(seat_account)
# print(seat_account, oa_id)  # 18
# for ans_ids in KG_2.keys([oa_id]):
#     if "车牌" in KG_2[ans_ids]:
#         print(KG_2[ans_ids])
# exit()

# # 测试解码流程
# KG.load("data/KG.json")
# model.load_weights("./best_model.weights")  # best_model.e15.weights
# print(autoqa.generate(["", "", "我在天街消费后珑珠没有到账"], topk=3))
# exit()


if __name__ == "__main__":

    # datafile = "data/C2ZXKF坐席对话数据20211001至20220301_训练数据_2022-04-22.pkl"
    datafile = "data/C2ZXKF坐席对话数据20211001至20220301_训练数据_2022-05-30.pkl"
    with codecs.open(datafile, "rb") as f:
        train_origin_data, test_data = pickle.load(f), pickle.load(f)

    print("读取数据集...")
    first_n_sentences = 3  # 当前会话的n句上文
    train_data, train_rich_text = load_data(train_origin_data)  # [[["x","x","y"], ""], ...]     ["", ...]
    assert len(train_data)==len(train_rich_text)
    test_data, test_rich_text = load_test_data(test_data)

    # 训练数据写入文件（用于检查），格式：(context, answer)
    with codecs.open("data/train.data.txt", mode="w", encoding="utf-8") as f:
        for line in train_data:
            f.write(str(line))
            f.write("\n")

    # 转换知识库
    if os.path.exists("data/KG.json"):
        print("加载 KG.json")
        KG_1.load("data/KG.json")
        KG_2.load("data/KG_2.json")
    else:
        print("构建 KG.json")
        # # 将测试数据也加入前缀树构建！！
        # all_data = train_data + test_data
        # all_rich_text = train_rich_text + test_rich_text
        # assert len(all_data)==len(all_rich_text)
        for j in range(len(train_data)):
            answer = train_data[j][-1]
            oa_id, answer_msg = train_rich_text[j]
            ids = tokenizer.encode(answer)[0][1:-1]
            ids = ids[:max_a_len]   # 限制回复的最大长度
            KG_1[ids] = answer_msg    # text → message
            # 从坐席 oa_id 作为前缀树分支的开始
            ids_2 = [oa_id] + ids
            KG_2[ids_2] = answer_msg
        KG_1.save("data/KG.json")
        KG_2.save("data/KG_2.json")
        all_data = []
        all_rich_text = []
    # exit()

    # # 注意：下列代码仅用于快速验证代码是否有 bug
    # epochs = 2
    # train_data = train_data[:1280]
    # test_data = test_data[:128]

    # 模型训练
    KG = KG_1
    evaluator = Evaluator()
    train_generator = data_generator(train_data, batch_size)
    model.fit(
        train_generator.forfit(),
        steps_per_epoch=len(train_generator),
        epochs=epochs,
        callbacks=[evaluator],
    )

    # 演示效果
    just_show()
    test_predict(test_data, "data/predict.json", topk=3)  # # 评测(狠耗时！)

else:

    print("加载前缀树字典...")
    KG_1.load("data/KG.json")
    KG_2.load("data/KG_2.json")

    print("加载权重...")
    model.load_weights("./best_model.weights")  # best_model.e15.weights
    # just_show()
