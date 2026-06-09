"""bm25_index.py —— 自实现的轻量 BM25 稀疏检索。

BM25 是经典的"关键词检索"算法：根据词频(TF)和逆文档频率(IDF)给
"文档-查询"打分。它擅长精确匹配具体词（人名、编号、专有名词），
正好和 bge-m3 这种"语义检索"互补，合起来做 hybrid 混合检索。
"""

import math
import re
import unicodedata
from collections import defaultdict

import jieba

_ALNUM = re.compile(r"[a-z0-9][a-z0-9_.@%+-]*")
_CJK_SPAN = re.compile(r"[\u4e00-\u9fff]+")
_URL = re.compile(r"https?://[^\s，。；、)）>]+")
_EMAIL = re.compile(r"[a-z0-9_.+-]+@[a-z0-9.-]+")
_TITLE = re.compile(r"《([^》]{2,40})》")
_TOKEN_SPLIT = re.compile(r"[\s,，。；;:：!！?？()（）\\[\\]【】<>《》\"'“”‘’|/\\\\]+")

_STOPWORDS = {
    "的", "了", "是", "在", "和", "与", "及", "或", "也", "有", "无", "为",
    "什么", "哪些", "哪个", "多少", "几个", "是否", "有没有", "一共", "目前",
    "当前", "最新", "最近", "信息学院", "上海科技大学", "上海", "科技", "大学",
}


def _normalize(text):
    return unicodedata.normalize("NFKC", text or "").lower()


def _cjk_ngrams(text, min_n=2, max_n=4):
    for span in _CJK_SPAN.findall(text):
        for n in range(min_n, max_n + 1):
            if len(span) < n:
                continue
            for i in range(len(span) - n + 1):
                yield span[i:i + n]


def tokenize(text):
    """把一段文本切成 BM25 token：jieba 词、精确实体和中文 n-gram。"""
    text = _normalize(text)
    tokens = []

    # URL/邮箱/书名号内容要完整保留，适合网址、教师邮箱、课程名等精确问答。
    tokens.extend(_URL.findall(text))
    tokens.extend(_EMAIL.findall(text))
    tokens.extend(t.strip() for t in _TITLE.findall(text) if t.strip())

    # 英文、数字、课程代码、邮箱局部等。
    tokens.extend(_ALNUM.findall(text))

    # jieba 搜索模式会额外给出适合检索的细粒度词。
    for tok in jieba.cut_for_search(text):
        tok = tok.strip()
        if not tok or tok in _STOPWORDS:
            continue
        for part in _TOKEN_SPLIT.split(tok):
            if part and part not in _STOPWORDS:
                tokens.append(part)

    # 短中文 n-gram 补齐 jieba 未切好的课程名、地址、机构名。
    tokens.extend(_cjk_ngrams(text))

    return [t for t in tokens if len(t) > 1 and t not in _STOPWORDS]


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
