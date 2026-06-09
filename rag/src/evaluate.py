"""evaluate.py —— 批量评测脚本（框架）。

读取测试集，对每道题分别跑“纯 dense”和“hybrid”，
把系统回答和检索来源记录下来，导出成表格（作业要求的列）。

测试集格式：rag/data/testset.jsonl，每行一题：
    {"query": "...", "gt_answer": "...", "type": "factual"}

输出：rag/outputs/test_results.csv（用 Excel 可直接打开；需要 .xlsx 时另存为即可）
列：query / gt_answer / sys_resp_dense / sys_resp_hybrid
   / is_correct_dense / is_correct_hybrid / sources_dense / sources_hybrid
其中两个 is_correct_* 列先留空，由人工（或后续判定脚本）填 0/1。
"""

import json
import time
import pandas as pd

from rag_pipeline import RAGPipeline

TESTSET_PATH = "rag/data/testset.jsonl"
OUTPUT_PATH = "rag/outputs/test_results.csv"


def load_testset(path):
    """读取 jsonl 测试集，返回题目字典列表。"""
    items = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def main():
    testset = load_testset(TESTSET_PATH)
    print(f"加载 {len(testset)} 道测试题")

    # 加载一次 BM25，用开关切换 dense/hybrid
    rag = RAGPipeline(top_k=5, use_hybrid=True)

    rows = []
    for i, item in enumerate(testset):
        q = item["query"]
        print(f"[{i + 1}/{len(testset)}] {q}")

        # —— 纯 dense ——
        t0 = time.time()
        ans_dense, ctx_dense = rag.answer(q, hybrid=False)
        t_dense = time.time() - t0

        # —— hybrid：dense + BM25 ——
        t0 = time.time()
        ans_hybrid, ctx_hybrid = rag.answer(q, hybrid=True)
        t_hybrid = time.time() - t0

        rows.append({
            "query": q,
            "gt_answer": item.get("gt_answer", ""),
            "sys_resp_dense": ans_dense,
            "sys_resp_hybrid": ans_hybrid,
            "is_correct_dense": "",   # 待判定，填 0/1
            "is_correct_hybrid": "",  # 待判定，填 0/1
            "type": item.get("type", ""),
            "latency_dense_s": round(t_dense, 2),
            "latency_hybrid_s": round(t_hybrid, 2),
            "sources_dense": " | ".join(c["title"] for c in ctx_dense),
            "sources_hybrid": " | ".join(c["title"] for c in ctx_hybrid),
        })

    # 按作业要求的顺序排列前 6 列，其余附在后面备用
    columns = [
        "query", "gt_answer",
        "sys_resp_dense", "sys_resp_hybrid",
        "is_correct_dense", "is_correct_hybrid",
        "type", "latency_dense_s", "latency_hybrid_s",
        "sources_dense", "sources_hybrid",
    ]
    df = pd.DataFrame(rows, columns=columns)
    df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")
    print(f"✓ 已导出 {len(rows)} 行到 {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
