class LLMClient:
    def __init__(self, base_url: str, model_name: str):
        self.base_url = base_url
        self.model_name = model_name

    def call_judge(self, prompt: str) -> dict:
        """Wrapper OpenAI SDK gọi về LM Studio, xử lý retry, đo latency"""
        pass
