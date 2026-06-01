"""evaluate.py —— 批量评测脚本（框架）。

读取测试集，对每道题分别跑"优化前(无 rerank)"和"优化后(有 rerank)"，
把系统回答和检索来源记录下来，导出成表格（作业要求的列）。

测试集格式：rag/data/testset.jsonl，每行一题：
    {"query": "...", "gt_answer": "...", "type": "factual"}

输出：rag/outputs/test_results.csv（用 Excel 可直接打开；需要 .xlsx 时另存为即可）
列：query / gt_answer / sys_resp_before_opt / sys_resp_after_opt
   / is_correct_before_opt / is_correct_after_opt / sources_before / sources_after
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

    # 加载一次（含 reranker + BM25），用开关切换优化前/后
    rag = RAGPipeline(top_k=5, use_rerank=True, use_hybrid=True)

    rows = []
    for i, item in enumerate(testset):
        q = item["query"]
        print(f"[{i + 1}/{len(testset)}] {q}")

        # —— 优化前：纯 dense ——
        t0 = time.time()
        ans_before, ctx_before = rag.answer(q, rerank=False, hybrid=False)
        t_before = time.time() - t0

        # —— 优化后：hybrid + rerank ——
        t0 = time.time()
        ans_after, ctx_after = rag.answer(q, rerank=True, hybrid=True)
        t_after = time.time() - t0

        rows.append({
            "query": q,
            "gt_answer": item.get("gt_answer", ""),
            "sys_resp_before_opt": ans_before,
            "sys_resp_after_opt": ans_after,
            "is_correct_before_opt": "",   # 待判定，填 0/1
            "is_correct_after_opt": "",    # 待判定，填 0/1
            "type": item.get("type", ""),
            "latency_before_s": round(t_before, 2),
            "latency_after_s": round(t_after, 2),
            "sources_before": " | ".join(c["title"] for c in ctx_before),
            "sources_after": " | ".join(c["title"] for c in ctx_after),
        })

    # 按作业要求的顺序排列前 6 列，其余附在后面备用
    columns = [
        "query", "gt_answer",
        "sys_resp_before_opt", "sys_resp_after_opt",
        "is_correct_before_opt", "is_correct_after_opt",
        "type", "latency_before_s", "latency_after_s",
        "sources_before", "sources_after",
    ]
    df = pd.DataFrame(rows, columns=columns)
    df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")
    print(f"✓ 已导出 {len(rows)} 行到 {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
