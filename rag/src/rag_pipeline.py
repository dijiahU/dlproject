"""rag_pipeline.py —— 完整 RAG 流程：检索相关资料 + 让大模型基于资料作答。"""

from retriever import Retriever
from generator import Generator


class RAGPipeline:
    """把检索器和生成器组装在一起，对外提供一个 answer() 入口。"""

    def __init__(self, top_k=5, use_rerank=False, use_hybrid=False, gpu_memory_utilization=0.6):
        self.retriever = Retriever(use_rerank=use_rerank, use_hybrid=use_hybrid)
        self.generator = Generator(gpu_memory_utilization=gpu_memory_utilization)
        self.top_k = top_k

    def answer(self, question, rerank=None, hybrid=None):
        """rerank/hybrid=None 用实例默认；传 True/False 可临时切换优化前/后。"""
        contexts = self.retriever.search(question, top_k=self.top_k, rerank=rerank, hybrid=hybrid)
        answer = self.generator.generate(question, contexts)
        return answer, contexts


if __name__ == "__main__":
    # 同时开 rerank + hybrid，用开关对比"优化前(全关) vs 优化后(全开)"
    rag = RAGPipeline(top_k=5, use_rerank=True, use_hybrid=True)

    q = "计算机科学与技术专业需要修满多少学分才能毕业？"
    for name, kw in [("优化前(纯dense)", dict(rerank=False, hybrid=False)),
                     ("优化后(hybrid+rerank)", dict(rerank=True, hybrid=True))]:
        answer, contexts = rag.answer(q, **kw)
        print("\n" + "=" * 60)
        print(f"问题: {q}   （{name}）")
        print("-" * 60)
        print("回答:", answer)
        print("来源:", [c["title"] for c in contexts])
