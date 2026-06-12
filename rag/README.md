# 上海科技大学 / 信息学院 RAG 问答系统

本目录实现了一个面向上海科技大学及信息科学与技术学院（SIST）内容的中文 RAG 问答系统。当前版本使用本地开源模型推理，不依赖商业 LLM API；检索只保留 `dense` 和 `hybrid` 两种配置，已删除 cross-encoder rerank 相关代码与产物。

## 1. 当前架构

```
用户问题
  -> Retriever
       dense: BAAI/bge-m3 + FAISS
       hybrid: dense + jieba BM25 + RRF 融合
  -> Generator
       Qwen3.5-4B / Qwen3.5-9B, vLLM 本地推理
  -> 答案 + 来源片段
```

- 知识库：本次评测使用本地清洗版 `rag/data/chunks.jsonl`，共 253,032 段可检索文本。
- 向量索引：`rag/index/faiss.index` + `rag/index/meta.jsonl`。
- 测试集：`rag/data/testset.jsonl`，50 道题，字段为 `query`、`gt_answer`、`type`。
- 最终评测产物：`rag/outputs/eval_4b.json` 和 `rag/outputs/eval_9b.json`。

## 2. 目录结构

```
rag/
├── data/
│   ├── chunks.jsonl
│   ├── documents.jsonl
│   └── testset.jsonl
├── index/
│   ├── faiss.index
│   └── meta.jsonl
├── outputs/
│   ├── eval_4b.json
│   └── eval_9b.json
├── src/
│   ├── bm25_index.py
│   ├── build_index.py
│   ├── evaluate.py
│   ├── evaluate_compare.py
│   ├── generator.py
│   ├── rag_pipeline.py
│   └── retriever.py
├── assets/gradio_demo.png
├── report.md
└── requirements.txt
```

`index/` 和 `outputs/` 是运行产物。当前只保留最终人工核对过的 `eval_4b.json`、`eval_9b.json`；CSV 和 accuracy summary 都可以由脚本或 JSON 重新生成，不作为最终必需文件保留。

## 3. 环境准备

命令从仓库根目录 `<repo-root>` 执行。

```bash
conda create -n rag python=3.11 -y
conda activate rag
pip install -r rag/requirements.txt
```

注意：本次最新评测使用的清洗版大体积语料与索引文件不随提交上传。复现实验前请在本地准备 `rag/data/chunks.jsonl`、`rag/data/documents.jsonl`，然后重新运行索引构建。

集群上可以直接使用已有环境：

```bash
conda activate <rag-conda-env>
cd <repo-root>
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
```

生成模型由环境变量 `RAG_LLM` 指定。未设置时默认使用 `Qwen/Qwen3.5-4B`；跑 9B 时请改成真实 HuggingFace 模型名或本地模型目录，不能使用 `/path/to/your/9b/model` 这种占位路径。

## 4. 构建索引

知识库或 embedding 模型变化后需要重建索引：

```bash
python rag/src/build_index.py
```

输出文件为 `rag/index/faiss.index` 和 `rag/index/meta.jsonl`。`build_index.py` 与 `retriever.py` 中的 embedding 模型必须一致，当前为 `BAAI/bge-m3`。

## 5. 运行问答与 Gradio

命令行 smoke test：

```bash
python rag/src/retriever.py
python rag/src/generator.py
python rag/src/rag_pipeline.py
```

`rag_pipeline.py` 已内置 Gradio。默认启动 `0.0.0.0:7860`，默认使用 `top_k=8` 和 hybrid 检索，页面上可以取消勾选 hybrid 回到 dense。

服务器端在 GPU 节点上启动服务。下面是当前最佳 4B + hybrid Gradio 的通用启动方式：

```bash
conda activate <rag-conda-env>
cd <repo-root>
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

RAG_LLM=Qwen/Qwen3.5-4B \
RAG_GPU_UTIL=0.6 \
GRADIO_SERVER_NAME=0.0.0.0 \
GRADIO_SERVER_PORT=7860 \
python rag/src/rag_pipeline.py
```

如需测试 9B，将 `RAG_LLM` 改为 9B 的模型名或本地模型目录，并将 `RAG_GPU_UTIL` 调到如 `0.8`。

如果在本机浏览器访问集群服务，需要 SSH 转发。本地端口示例使用 `1786`，远端 Gradio 端口为 `7860`：

```bash
ssh -J <user>@<login-host> -N -L 1786:127.0.0.1:7860 <user>@<gpu-node>
```

这条 SSH 命令没有输出是正常的，保持该终端不要关闭。然后在本机浏览器打开 `http://127.0.0.1:1786`。如果服务跑在别的 GPU 节点，把 `<gpu-node>` 改成对应节点；如果本地 `1786` 被占用，换成其他本地端口即可。

## 6. 批量评测

基础评测脚本会对每题分别运行 dense 和 hybrid，并导出 CSV，正确性列需要人工填写。该 CSV 是临时评测表，不是最终归档产物：

```bash
python rag/src/evaluate.py
```

对比评测脚本支持通过环境变量指定模型和输出路径：

```bash
RAG_LLM=/真实/4b/模型/路径 \
RAG_GPU_UTIL=0.6 \
RAG_OUT=/tmp/test_results_current_4b.csv \
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
python rag/src/evaluate_compare.py
```

```bash
RAG_LLM=/真实/9b/模型/路径 \
RAG_GPU_UTIL=0.8 \
RAG_OUT=/tmp/test_results_current_9b.csv \
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
python rag/src/evaluate_compare.py
```

当前最终结果已经人工逐题核对并整合到：

- `rag/outputs/eval_4b.json`
- `rag/outputs/eval_9b.json`



## 7. 当前评测结论

当前生成 prompt 为 v2：允许模型在资料证据充分时进行合并、计数、日期筛选、比较和二跳推理；只有完全没有相关证据或证据冲突无法判断时才拒答。50 题人工核对结果：

| 模型 | dense | hybrid |
|---|---:|---:|
| Qwen3.5-4B | 30 / 50（60.00%） | 35 / 50（70.00%） |
| Qwen3.5-9B | 30 / 50（60.00%） | 33 / 50（66.00%） |

当前最佳配置为 **Qwen3.5-4B + hybrid，不使用 rerank**。9B 在 prompt v2 下从 30/50 提升到 33/50，但仍更容易保守拒答或给出多版本解释。rerank 在本测试集中会降低候选多样性，对列表题、多来源题和二跳证据题不稳定，因此已经删除。

## 8. 已知局限

- 当前语料虽已扩展到 253,032 个 chunk，但学校主站、组织结构、最新招生通知、校徽、学术活动等全校层面证据仍不完整。
- “最新/最近/截至某日”问题依赖发布时间排序和最新网页覆盖，当前检索只能部分处理。
- 列表题和多跳题需要多个页面共同支撑，单轮 top-k 容易遗漏关键片段。
- 部分答案字符串在语料中有无关命中，但缺少正确上下文或关系，不能视为可回答证据。详见 `rag/report.md` 的“主要发现”。

## 9. 参考

- BGE-M3: https://huggingface.co/BAAI/bge-m3
- Qwen: https://huggingface.co/Qwen
- FAISS: https://github.com/facebookresearch/faiss
- Sentence-Transformers: https://www.sbert.net/
- vLLM: https://github.com/vllm-project/vllm
- Gradio: https://www.gradio.app/
