# 上海科技大学 / 信息学院 RAG 问答系统

本仓库实现了一个面向上海科技大学及信息科学与技术学院（SIST）内容的中文 RAG 问答系统。当前版本使用本地开源模型推理，不依赖商业 LLM API；检索只保留 `dense` 和 `hybrid` 两种配置，已删除 cross-encoder rerank 相关代码与产物。

详细运行说明见 [rag/README.md](rag/README.md)。

## 快速入口

- 代码目录：`rag/src/`
- 知识库：`rag/data/chunks.jsonl`
- 测试集：`rag/data/testset.jsonl`
- 最终评测结果：`rag/outputs/eval_4b.json`、`rag/outputs/eval_9b.json`
- 初步评测报告：`rag/report.md`
- Gradio 截图：`rag/assets/gradio_demo.png`

## 当前评测结果

50 题人工核对结果：

| 模型 | dense | hybrid |
|---|---:|---:|
| Qwen3.5-4B | 15 / 50（30.00%） | 15 / 50（30.00%） |
| Qwen3.5-9B | 20 / 50（40.00%） | 29 / 50（58.00%） |

推荐配置为 **Qwen3.5-9B + hybrid，不使用 rerank**。

## Gradio 快速启动

在 GPU 节点上启动服务：

```bash
conda activate <rag-conda-env>
cd <repo-root>
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

RAG_LLM=Qwen/Qwen3.5-9B \
RAG_GPU_UTIL=0.8 \
GRADIO_SERVER_NAME=0.0.0.0 \
GRADIO_SERVER_PORT=7860 \
python rag/src/rag_pipeline.py
```

本机浏览器访问集群服务时，用 SSH 转发：

```bash
ssh -J <user>@<login-host> -N -L 1786:127.0.0.1:7860 <user>@<gpu-node>
```

然后打开 `http://127.0.0.1:1786`。

## 目录说明

`rag/` 保留为 RAG 子项目目录，避免把数据、索引、报告、输出和源码全部平铺到仓库根目录。现有代码中的默认路径也都以 `rag/` 为前缀，因此保留该结构可以减少无关路径改动。
