"""rag_pipeline.py —— 完整 RAG 流程：检索相关资料 + 让大模型基于资料作答。"""

import os

import gradio as gr

from retriever import Retriever
from generator import Generator


class RAGPipeline:
    """把检索器和生成器组装在一起，对外提供一个 answer() 入口。"""

    def __init__(self, top_k=5, use_hybrid=False, gpu_memory_utilization=None):
        # 显存占用可由环境变量 RAG_GPU_UTIL 覆盖（跑 9B 时调大，如 0.8）
        if gpu_memory_utilization is None:
            gpu_memory_utilization = float(os.environ.get("RAG_GPU_UTIL", "0.6"))
        self.retriever = Retriever(use_hybrid=use_hybrid)
        self.generator = Generator(gpu_memory_utilization=gpu_memory_utilization)
        self.top_k = top_k

    def answer(self, question, hybrid=None):
        """hybrid=None 用实例默认；传 True/False 可临时切换 dense/hybrid。"""
        contexts = self.retriever.search(question, top_k=self.top_k, hybrid=hybrid)
        answer = self.generator.generate(question, contexts)
        return answer, contexts


def format_contexts(contexts):
    """把检索到的来源整理成适合 Gradio 展示的文本。"""
    if not contexts:
        return "无可用来源。"

    lines = []
    for i, c in enumerate(contexts, 1):
        lines.append(f"{i}. {c['title']}")
        lines.append(f"   URL: {c['url']}")
        snippet = c["text"].strip().replace("\n", " ")
        if len(snippet) > 300:
            snippet = snippet[:300] + "..."
        lines.append(f"   摘要: {snippet}")
    return "\n".join(lines)


def build_gradio_demo(rag):
    """构建 Gradio 界面。"""
    def ask(question, hybrid):
        question = (question or "").strip()
        if not question:
            return "请输入问题。", "无可用来源。"

        answer, contexts = rag.answer(question, hybrid=hybrid)
        return answer, format_contexts(contexts)

    return gr.Interface(
        fn=ask,
        inputs=[
            gr.Textbox(label="问题", placeholder="请输入你的问题"),
            gr.Checkbox(label="启用 hybrid", value=True),
        ],
        outputs=[
            gr.Textbox(label="回答", lines=10),
            gr.Textbox(label="来源", lines=12),
        ],
        title="RAG 问答系统",
        description="输入问题，选择是否启用 hybrid 检索。",
    )


if __name__ == "__main__":
    gpu_util = float(os.environ.get("RAG_GPU_UTIL", "0.8"))
    server_name = os.environ.get("GRADIO_SERVER_NAME", "0.0.0.0")
    server_port = int(os.environ.get("GRADIO_SERVER_PORT", "7860"))
    share = os.environ.get("GRADIO_SHARE", "0") == "1"

    # 默认使用当前评测最优的 hybrid 配置；界面上仍可取消勾选回到 dense。
    rag = RAGPipeline(top_k=8, use_hybrid=True, gpu_memory_utilization=gpu_util)
    demo = build_gradio_demo(rag)
    demo.launch(server_name=server_name, server_port=server_port, share=share)
