# 上海科技大学 / 信息学院（SIST）RAG 问答系统

一个基于检索增强生成（Retrieval-Augmented Generation, RAG）的中文问答系统，能够回答关于上海科技大学及信息科学与技术学院（SIST）的问题。系统**完全使用本地/自部署的开源模型**，不依赖任何商业 LLM API。

---

## 1. 系统架构

```
用户问题
   │
   ▼
┌─────────────────── 检索 Retriever ───────────────────┐
│  ① 召回阶段                                            │
│     · Dense：bge-m3 编码 + FAISS 向量检索              │
│     · （可选）Hybrid：Dense + BM25，用 RRF 融合         │
│  ② 精排阶段（可选）                                    │
│     · Reranker（bge-reranker-v2-m3）对候选交叉打分重排  │
└───────────────────────────────────────────────────────┘
   │  top-k 相关片段
   ▼
┌─────────────────── 生成 Generator ───────────────────┐
│  Qwen3.5-4B (vLLM)，按 prompt 模板基于检索内容作答      │
└───────────────────────────────────────────────────────┘
   │
   ▼
答案 + 参考来源
```

- **知识库**：约 12,533 段已清洗、切分的文本（`chunks.jsonl`），来源为 SIST 官网及子域名。
- **两个可开关的优化**：Reranking、Hybrid 混合检索（详见第 6 节）。

---

## 2. 主要组件

| 组件 | 说明 |
|------|------|
| Embedding | `BAAI/bge-m3`（多语言，1024 维） |
| 向量索引 | FAISS `IndexFlatIP`（内积=余弦，因向量已归一化） |
| 稀疏检索 | 自实现 BM25（`bm25_index.py`，零依赖） |
| Reranker | `BAAI/bge-reranker-v2-m3`（交叉编码器） |
| 生成模型 | `Qwen/Qwen3.5-4B`（vLLM 推理，`language_model_only`） |

---

## 3. 目录结构

```
rag/
├── data/
│   ├── chunks.jsonl        # 知识库（每行一段可检索文本）
│   ├── documents.jsonl     # 文档级元数据
│   └── testset.jsonl       # 测试集（query + gt_answer + type）
├── src/
│   ├── build_index.py      # 建索引：编码全部 chunk → FAISS + meta
│   ├── bm25_index.py       # 自实现 BM25（hybrid 用）
│   ├── retriever.py        # 检索器：dense / hybrid / rerank
│   ├── generator.py        # 生成器：Qwen (vLLM)
│   ├── rag_pipeline.py     # 完整流程：检索 + 生成
│   └── evaluate.py         # 批量评测，导出 CSV
│   # 注：网页界面(Gradio app.py)由界面/部署同学负责，不在本模块内
├── index/                  # 生成物：faiss.index + meta.jsonl
├── outputs/                # 生成物：test_results.csv
└── README.md
```

> 注：`index/` 和 `outputs/` 为运行生成，提交时不必包含；原始大数据与模型 checkpoint 也不应放入提交包。

---

## 4. 环境准备

### 4.1 依赖

见 `requirements.txt`。核心依赖：`torch`、`sentence-transformers`、`faiss-cpu`、`vllm`、`transformers`、`pandas`。BM25 为自实现，无需额外依赖。

**在你自己的机器上配置同样的环境（推荐用 conda，需自备 NVIDIA GPU）：**

```bash
conda create -n rag python=3.11 -y
conda activate rag
pip install -r requirements.txt
```

> - **GPU 自理**：vLLM 跑 Qwen 需要一张 NVIDIA 显卡（建议 ≥24GB 显存；显存小可改用更小模型，见 4.2）。`torch`/`vllm` 装的是 CUDA 版本，请确保本机 CUDA 驱动匹配。
> - `requirements.txt` 里的版本是集群上测试通过的版本；若你的 CUDA/硬件不同，`torch`、`vllm` 可装与自己 CUDA 匹配的版本，其余保持即可。
> - 本项目最初在校内 SLURM 集群的现成 conda 环境上开发（`conda activate /home/.../askbench-qwen35`），那是集群专用、与本机部署无关，按上面自己建环境即可。

### 4.2 模型

首次运行会从 HuggingFace 自动下载（约共 12GB），需联网。国内网络慢时可走镜像：

```bash
export HF_ENDPOINT=https://hf-mirror.com
```

需要的模型：

- `BAAI/bge-m3`（embedding）、`BAAI/bge-reranker-v2-m3`（rerank）、`Qwen/Qwen3.5-4B`（生成）

显存不足时，把 `generator.py` 里的 `MODEL_NAME` 换成更小的 Qwen（如 `Qwen/Qwen2.5-3B-Instruct` 等本机能跑的开源指令模型）即可，其余代码不用改。

模型已下载到本地缓存后，可设离线模式避免每次联网检查：

```bash
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
```

### 4.3 GPU

生成（vLLM 跑 Qwen）和建索引需要 NVIDIA GPU，**显卡由各自机器自备**。三个模型共卡时已用 `gpu_memory_utilization=0.6` 控制占用，约 24–40GB 显存即可；显存更小就换更小的生成模型。

> （仅校内 SLURM 集群相关，本机部署可忽略）申请交互式 GPU：`srun -p gpu --gres=gpu:1 -c 8 --mem=32G -t 02:00:00 --pty bash`；计算节点外网受限时 `unset http_proxy https_proxy ...` 并设离线变量。

---

## 5. 运行步骤

所有命令在项目根目录 `dlproject/`（即 `rag/` 的上一级）下执行。

### 5.1 构建索引（只需一次）

```bash
python rag/src/build_index.py
```

- 默认 `MAX_CHUNKS=None` 表示编码全部 12,533 段；可改为整数做小批量测试。
- 输出：`rag/index/faiss.index` 与 `rag/index/meta.jsonl`。
- GPU 上约 1–3 分钟；CPU 上约 50 分钟。

> ⚠️ 重建索引的时机：知识库变化或更换 embedding 模型时。`build_index.py` 与 `retriever.py` 的 `MODEL_NAME` 必须一致。

### 5.2 单独测试各组件

```bash
python rag/src/retriever.py     # 对比 dense / +rerank / hybrid / 全开 的检索结果
python rag/src/generator.py     # 用假资料测试生成
python rag/src/rag_pipeline.py  # 完整问答（优化前 vs 优化后）
```

### 5.3 批量评测

```bash
python rag/src/evaluate.py
```

读取 `rag/data/testset.jsonl`，对每题分别跑“优化前/优化后”，导出 `rag/outputs/test_results.csv`（`utf-8-sig`，Excel 可直接打开）。列：`query, gt_answer, sys_resp_before_opt, sys_resp_after_opt, is_correct_before_opt, is_correct_after_opt`（后两列正确性需人工标注 0/1），另附题型、延迟、检索来源。

> ⚠️ **关于测试集（重要）**：当前 `rag/data/testset.jsonl` 里只有 **7 道占位题、且标准答案是占位的**，仅用于验证评测流程能跑通，**不能用来下"优化有没有用"的结论**。正式测评需要负责评测的同学**自己出题**：
> 1. 按作业要求写 **≥50 道**问题，覆盖多种题型（事实 factual / 多跳 multi-hop / 时效 time-sensitive / 比较 comparative / 条件 conditional），每行一题，格式：
>    ```json
>    {"query": "问题", "gt_answer": "对照真实数据写的标准答案", "type": "factual"}
>    ```
> 2. 跑 `python rag/src/evaluate.py` 生成两列系统回答；
> 3. **人工逐题判定**正确性，把 `is_correct_before_opt` / `is_correct_after_opt` 填 0/1；
> 4. 即可统计"优化前 vs 优化后"准确率（建议再按题型分组分析），填入报告与提交用 Excel。

---

## 6. 两个优化（可开关，便于做“优化前后”对比）

检索行为由 `Retriever.search(query, top_k, rerank, hybrid)` 或 `RAGPipeline.answer(question, rerank, hybrid)` 控制：

| 配置 | rerank | hybrid | 说明 |
|------|--------|--------|------|
| Baseline | False | False | 纯 dense 检索 top-k |
| Reranking | True | False | 召回 20 → reranker 精排 → top-k |
| Hybrid | False | True | dense + BM25，RRF 融合 → top-k |
| 全开 | True | True | hybrid 召回 → reranker 精排 |

- **Reranking**：用交叉编码器在“问题+片段”成对输入上精细打分，提升相关性。`recall_k=20` 控制召回池大小。
- **Hybrid**：Dense（语义）+ BM25（关键词）互补，用 **RRF（倒数排名融合，`RRF_K=60`）** 合并两路排名，对含具体数字/专有名词的问题更稳。

示例（Python）：

```python
from rag_pipeline import RAGPipeline
rag = RAGPipeline(top_k=5, use_rerank=True, use_hybrid=True)
ans_before, _ = rag.answer("计算机专业要修多少学分毕业？", rerank=False, hybrid=False)
ans_after,  _ = rag.answer("计算机专业要修多少学分毕业？", rerank=True,  hybrid=True)
```

---

## 7. 已知局限

- **多跳问题**：单轮检索难以“先查 A 再查 B”，可能答不全。
- **时效性问题**：检索不带时间排序，“最新”类问题不保证返回日期最新内容。
- **数据质量**：部分 chunk 标题为空（约 2%）或含网页噪音/编码损坏；可通过数据清洗改善。
- **覆盖范围**：提供的数据集以 SIST 为主，全校层面问题可能缺乏依据。

---

## 8. 模型与工具引用

- BGE-M3 / BGE-Reranker：BAAI（https://huggingface.co/BAAI）
- Qwen：阿里通义千问（https://huggingface.co/Qwen）
- FAISS：https://github.com/facebookresearch/faiss
- Sentence-Transformers：https://www.sbert.net/
- vLLM：https://github.com/vllm-project/vllm
- Gradio：https://www.gradio.app/
