"""generator.py —— 用本地 Qwen(vLLM) 根据检索到的资料生成答案。"""

import os

from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

# 可用环境变量 RAG_LLM 覆盖模型（如指向本地 9B 路径），默认 4B
MODEL_NAME = os.environ.get("RAG_LLM", "Qwen/Qwen3.5-4B")
PROMPT_TOKEN_MARGIN = 64

# 系统提示词：规定模型的身份和行为准则（尤其"不许编造"很关键）
SYSTEM_PROMPT = (
    "你是上海科技大学的智能问答助手。"
    "请严格根据提供的【资料】回答【问题】；"
    "若资料中没有相关信息，请回答“根据现有资料无法回答”，不要编造。"
)

# 用户提示词模板：把资料和问题填进固定格式
PROMPT_TEMPLATE = """【资料】
{context}

【问题】{question}"""


class Generator:
    """生成器：内部持有 vLLM 引擎和分词器，对外提供 generate() 方法。"""

    def __init__(self, model_name=MODEL_NAME, max_model_len=12288, gpu_memory_utilization=0.9):
        print("正在加载大模型（vLLM 首次加载需一两分钟）...")
        self._validate_model_name(model_name)
        self.max_model_len = max_model_len
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.llm = LLM(
            model=model_name,
            dtype="bfloat16",          # 用 bf16 精度，省显存、速度快
            max_model_len=max_model_len,
            gpu_memory_utilization=gpu_memory_utilization,
            language_model_only=True,  # 这些模型是多模态的，这里只用语言部分
        )
        print("大模型就绪")

    @staticmethod
    def _validate_model_name(model_name):
        """Catch common placeholder/path mistakes before Transformers emits a long stack trace."""
        if not model_name:
            raise ValueError("RAG_LLM 不能为空；请设置为 HuggingFace 模型名或本地模型目录。")
        if "/path/to/your" in model_name:
            raise ValueError(
                "RAG_LLM 仍是示例占位符。请改成真实模型名或本地目录，例如 "
                "`Qwen/Qwen3.5-9B` 或 `/home/hcj/models/Qwen3.5-9B`。"
            )
        if model_name.startswith("/") and not os.path.exists(model_name):
            raise FileNotFoundError(
                f"RAG_LLM 指向的本地模型目录不存在：{model_name}"
            )

    def _format_prompt(self, question, context):
        user_content = PROMPT_TEMPLATE.format(context=context, question=question)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]
        return self.tokenizer.apply_chat_template(
            messages, tokenize=False,
            add_generation_prompt=True, enable_thinking=False,
        )

    def _token_count(self, text):
        return len(self.tokenizer.encode(text, add_special_tokens=False))

    def _decode_token_prefix(self, text, max_tokens):
        token_ids = self.tokenizer.encode(text, add_special_tokens=False)
        if len(token_ids) <= max_tokens:
            return text
        return self.tokenizer.decode(token_ids[:max_tokens], skip_special_tokens=True)

    def _build_prompt(self, question, contexts, max_new_tokens):
        """Build a chat prompt that fits the model context window."""
        max_prompt_tokens = self.max_model_len - max_new_tokens - PROMPT_TOKEN_MARGIN
        if max_prompt_tokens <= 0:
            raise ValueError(
                f"max_model_len={self.max_model_len} 太小，无法预留 {max_new_tokens} 个输出 token。"
            )

        full_context = "\n\n".join(
            f"[资料{i + 1}] {c['text']}" for i, c in enumerate(contexts)
        )
        prompt = self._format_prompt(question, full_context)
        if self._token_count(prompt) <= max_prompt_tokens:
            return prompt

        empty_prompt = self._format_prompt(question, "")
        context_budget = max_prompt_tokens - self._token_count(empty_prompt)
        if context_budget <= 0:
            return empty_prompt

        parts = []
        remaining = context_budget
        for i, c in enumerate(contexts):
            piece = f"[资料{i + 1}] {c['text']}"
            piece_tokens = self._token_count(piece)
            if piece_tokens <= remaining:
                parts.append(piece)
                remaining -= piece_tokens
                continue
            if remaining > 32:
                parts.append(self._decode_token_prefix(piece, remaining))
            break

        trimmed_prompt = self._format_prompt(question, "\n\n".join(parts))
        while self._token_count(trimmed_prompt) > max_prompt_tokens and parts:
            parts[-1] = self._decode_token_prefix(parts[-1], max(32, int(self._token_count(parts[-1]) * 0.85)))
            trimmed_prompt = self._format_prompt(question, "\n\n".join(parts))
        return trimmed_prompt

    def generate(self, question, contexts, max_new_tokens=512):
        # 1) 套用 Qwen 聊天模板，并按 token 预算裁剪长资料，避免超过 vLLM 上下文。
        prompt = self._build_prompt(question, contexts, max_new_tokens)
        # 2) 设定采样参数：temperature=0 表示"每次都给最确定的答案"
        sampling = SamplingParams(temperature=0.0, max_tokens=max_new_tokens)
        # 3) 生成！
        outputs = self.llm.generate([prompt], sampling)
        text = outputs[0].outputs[0].text
        # 4) 去掉可能残留的空"思考块"
        if "</think>" in text:
            text = text.split("</think>", 1)[1]
        return text.strip()


if __name__ == "__main__":
    gen = Generator()

    # 先用"假资料"测试生成器本身（暂时不接检索，单独验证这一块）
    fake_contexts = [
        {"text": "上海科技大学信息科学与技术学院(SIST)成立于2013年。"},
        {"text": "信息学院研究方向包括：视觉与数据智能、智能网络、后摩尔芯片、智慧电气、机器人、系统安全、智能医学。"},
    ]
    answer = gen.generate("信息学院有哪些研究方向？", fake_contexts)
    print("\n=== 模型回答 ===")
    print(answer)
