"""generator.py —— 用本地 Qwen(vLLM) 根据检索到的资料生成答案。"""

from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

MODEL_NAME = "Qwen/Qwen3.5-4B"

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

    def __init__(self, model_name=MODEL_NAME, max_model_len=8192, gpu_memory_utilization=0.9):
        print("正在加载大模型（vLLM 首次加载需一两分钟）...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.llm = LLM(
            model=model_name,
            dtype="bfloat16",          # 用 bf16 精度，省显存、速度快
            max_model_len=max_model_len,
            gpu_memory_utilization=gpu_memory_utilization,
            language_model_only=True,  # 这些模型是多模态的，这里只用语言部分
        )
        print("大模型就绪")

    def generate(self, question, contexts, max_new_tokens=512):
        # 1) 把多段资料拼成一整块文本
        context = "\n\n".join(
            f"[资料{i + 1}] {c['text']}" for i, c in enumerate(contexts)
        )
        # 2) 填进模板，得到用户消息内容
        user_content = PROMPT_TEMPLATE.format(context=context, question=question)
        # 3) 组装成"对话"格式（系统 + 用户）
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]
        # 4) 套用 Qwen 的聊天模板，变成模型真正吃的字符串
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False,
            add_generation_prompt=True, enable_thinking=False,
        )
        # 5) 设定采样参数：temperature=0 表示"每次都给最确定的答案"
        sampling = SamplingParams(temperature=0.0, max_tokens=max_new_tokens)
        # 6) 生成！
        outputs = self.llm.generate([prompt], sampling)
        text = outputs[0].outputs[0].text
        # 7) 去掉可能残留的空"思考块"
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
