"""retriever.py —— 检索器：dense(bge-m3+FAISS) / hybrid(+BM25) / 可选 rerank。"""

import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer, CrossEncoder

from bm25_index import BM25, tokenize

MODEL_NAME = "BAAI/bge-m3"
INDEX_PATH = "rag/index/faiss.index"
META_PATH  = "rag/index/meta.jsonl"
RERANKER_NAME = "BAAI/bge-reranker-v2-m3"

RRF_K = 60          # RRF 融合常数（经验值，越大越"平均"）


def load_meta(path):
    """读取 meta.jsonl，返回字典列表（顺序与索引里的向量一一对应）。"""
    meta = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                meta.append(json.loads(line))
    return meta


class Retriever:
    def __init__(self, model_name=MODEL_NAME, index_path=INDEX_PATH, meta_path=META_PATH,
                 use_rerank=False, reranker_name=RERANKER_NAME, recall_k=20,
                 use_hybrid=False):
        print("正在加载模型和索引...")
        self.model = SentenceTransformer(model_name)
        self.index = faiss.read_index(index_path)
        self.meta = load_meta(meta_path)

        self.use_rerank = use_rerank
        self.use_hybrid = use_hybrid
        self.recall_k = recall_k
        self.reranker = CrossEncoder(reranker_name) if use_rerank else None

        # 仅在需要 hybrid 时才构建 BM25（对全部文本分词，约几秒~十几秒）
        self.bm25 = None
        if use_hybrid:
            print("构建 BM25 索引...")
            corpus = [tokenize(m["text"]) for m in self.meta]
            self.bm25 = BM25(corpus)

        print(f"就绪：{self.index.ntotal} 段文本 "
              f"(rerank={'开' if use_rerank else '关'}, hybrid={'开' if use_hybrid else '关'})")

    # ---------- 召回阶段 ----------
    def _dense_recall(self, query, n):
        """bge-m3 + FAISS 稠密召回，返回 [(idx, 余弦分), ...]。"""
        q_emb = self.model.encode([query], normalize_embeddings=True)
        q_emb = np.asarray(q_emb, dtype="float32")
        scores, indices = self.index.search(q_emb, n)
        return list(zip(indices[0].tolist(), scores[0].tolist()))

    def _hybrid_recall(self, query, n):
        """dense + BM25，用 RRF(倒数排名融合) 合并，返回 [(idx, 融合分), ...]。"""
        dense = self._dense_recall(query, n)
        bm25_top = self.bm25.top_n(tokenize(query), n)

        # 各自的"排名"（第几名，从 0 开始）
        dense_rank = {idx: r for r, (idx, _) in enumerate(dense)}
        bm25_rank = {idx: r for r, (idx, _) in enumerate(bm25_top)}

        # RRF：每个文档分数 = Σ 1/(K + 它在该路检索里的排名)
        fused = []
        for idx in set(dense_rank) | set(bm25_rank):
            s = 0.0
            if idx in dense_rank:
                s += 1.0 / (RRF_K + dense_rank[idx])
            if idx in bm25_rank:
                s += 1.0 / (RRF_K + bm25_rank[idx])
            fused.append((idx, s))
        fused.sort(key=lambda x: x[1], reverse=True)
        return fused[:n]

    # ---------- 对外接口 ----------
    def search(self, query, top_k=5, rerank=None, hybrid=None):
        """检索 top_k。rerank/hybrid 传 None 用实例默认，传 True/False 可临时覆盖。"""
        use_rerank = self.use_rerank if rerank is None else rerank
        use_hybrid = self.use_hybrid if hybrid is None else hybrid

        # 要重排或要融合，就多召回一些候选；否则直接取 top_k
        recall_n = self.recall_k if (use_rerank or use_hybrid) else top_k

        if use_hybrid:
            recalled = self._hybrid_recall(query, recall_n)
        else:
            recalled = self._dense_recall(query, recall_n)

        candidates = []
        for idx, score in recalled:
            item = self.meta[idx]
            candidates.append({
                "score": float(score),       # dense=余弦分 / hybrid=RRF融合分
                "title": item["title"],
                "url":   item["url"],
                "text":  item["text"],
            })

        if use_rerank:
            candidates = self.rerank(query, candidates)

        return candidates[:top_k]

    def rerank(self, query, candidates):
        """用 reranker 交叉编码器对候选重新打分并降序排列。"""
        pairs = [(query, c["text"]) for c in candidates]
        rerank_scores = self.reranker.predict(pairs)
        for c, s in zip(candidates, rerank_scores):
            c["rerank_score"] = float(s)
        candidates.sort(key=lambda c: c["rerank_score"], reverse=True)
        return candidates


if __name__ == "__main__":
    query = "计算机科学与技术专业需要修满多少学分才能毕业？"

    # 一个实例同时开 rerank + hybrid（加载 reranker、构建 BM25）
    retriever = Retriever(use_rerank=True, use_hybrid=True)

    configs = [
        ("① 纯 dense（baseline）",     dict(rerank=False, hybrid=False)),
        ("② dense + rerank",          dict(rerank=True,  hybrid=False)),
        ("③ hybrid(dense+BM25)",      dict(rerank=False, hybrid=True)),
        ("④ hybrid + rerank（全开）",  dict(rerank=True,  hybrid=True)),
    ]
    for name, kw in configs:
        print(f"\n===== {name} =====")
        for i, r in enumerate(retriever.search(query, top_k=3, **kw)):
            print(f"  {i + 1}. {r['title']}")
