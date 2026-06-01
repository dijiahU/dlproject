"""bm25_index.py —— 自实现的轻量 BM25 稀疏检索（零依赖）。

BM25 是经典的"关键词检索"算法：根据词频(TF)和逆文档频率(IDF)给
"文档-查询"打分。它擅长精确匹配具体词（人名、编号、专有名词），
正好和 bge-m3 这种"语义检索"互补，合起来做 hybrid 混合检索。
"""

import re
import math
from collections import defaultdict

# 简易分词：英文/数字按"词"切，中文按"单字"切（不依赖 jieba）
_EN = re.compile(r"[a-z0-9]+")
_CJK = re.compile(r"[一-鿿]")


def tokenize(text):
    """把一段文本切成 token 列表。"""
    text = text.lower()
    return _EN.findall(text) + _CJK.findall(text)


class BM25:
    """标准 BM25(Okapi)。用倒排索引加速打分。"""

    def __init__(self, corpus_tokens, k1=1.5, b=0.75):
        self.k1 = k1                      # 控制词频饱和速度
        self.b = b                        # 控制文档长度归一化强度
        self.doc_len = [len(d) for d in corpus_tokens]
        self.N = len(corpus_tokens)
        self.avgdl = sum(self.doc_len) / self.N if self.N else 0.0

        # 倒排索引：token -> {doc_id: 该词在该文档出现次数}
        self.inverted = defaultdict(dict)
        df = defaultdict(int)             # 每个 token 出现在多少篇文档里
        for doc_id, tokens in enumerate(corpus_tokens):
            tf = defaultdict(int)
            for t in tokens:
                tf[t] += 1
            for t, freq in tf.items():
                self.inverted[t][doc_id] = freq
                df[t] += 1

        # 预计算每个 token 的 IDF
        self.idf = {
            t: math.log((self.N - d + 0.5) / (d + 0.5) + 1.0)
            for t, d in df.items()
        }

    def get_scores(self, query_tokens):
        """返回 {doc_id: bm25分数}，只包含命中查询词的文档。"""
        scores = defaultdict(float)
        for t in query_tokens:
            if t not in self.inverted:
                continue
            idf = self.idf[t]
            for doc_id, freq in self.inverted[t].items():
                dl = self.doc_len[doc_id]
                denom = freq + self.k1 * (1 - self.b + self.b * dl / self.avgdl)
                scores[doc_id] += idf * (freq * (self.k1 + 1)) / denom
        return scores

    def top_n(self, query_tokens, n):
        """返回分数最高的 n 个 (doc_id, score)。"""
        scores = self.get_scores(query_tokens)
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return ranked[:n]
