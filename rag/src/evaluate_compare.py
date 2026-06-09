"""evaluate_compare.py —— 多检索配置对比评测。

默认对每题跑两种配置：dense / hybrid。
环境变量：
  RAG_LLM      指定生成模型（如 /home/hcj/models/Qwen3.5-9B），默认 4B
  RAG_GPU_UTIL vLLM 显存占用比例（跑 9B 建议 0.8），默认 0.6
  RAG_OUT      输出 CSV 路径，默认 rag/outputs/test_results_compare.csv
"""

import os
import json
import time
import pandas as pd

from rag_pipeline import RAGPipeline

TESTSET_PATH = "rag/data/testset.jsonl"
OUTPUT_PATH = os.environ.get("RAG_OUT", "rag/outputs/test_results_compare.csv")


def load_testset(path):
    items = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def main():
    testset = load_testset(TESTSET_PATH)
    print(f"加载 {len(testset)} 道测试题 | 模型={os.environ.get('RAG_LLM', 'Qwen/Qwen3.5-4B')}")

    rag = RAGPipeline(top_k=8, use_hybrid=True)

    configs = [
        ("dense", dict(hybrid=False)),
        ("hybrid", dict(hybrid=True)),
    ]

    rows = []
    for i, item in enumerate(testset):
        q = item["query"]
        print(f"[{i + 1}/{len(testset)}] {q}")
        row = {"query": q, "gt_answer": item.get("gt_answer", ""), "type": item.get("type", "")}
        for name, kw in configs:
            t0 = time.time()
            ans, _ = rag.answer(q, **kw)
            row[f"resp_{name}"] = ans
            row[f"lat_{name}"] = round(time.time() - t0, 2)
        rows.append(row)
        pd.DataFrame(rows).to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")

    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")
    print(f"✓ 已导出 {len(rows)} 行到 {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
