"""
Use GPT Series Models (supports both OpenAI & Azure OpenAI)
"""

import time
from openai import OpenAI, AzureOpenAI

class GPT():
    def __init__(self, model, base_url, api_key, api_version="2025-01-01-preview"):
        self.model_name = model
        self.base_url = base_url
        self.api_key = api_key
        self.api_version = api_version
        self._init_model()

    def _init_model(self):
        """
        Initialize OpenAI or AzureOpenAI client automatically
        """
        if "azure" in self.base_url:
            print("✅ Using Azure OpenAI endpoint")
            self.client = AzureOpenAI(
                azure_endpoint=self.base_url.rstrip("/"),
                api_key=self.api_key,
                api_version=self.api_version
            )
        else:
            print("✅ Using OpenAI standard endpoint")
            self.client = OpenAI(
                base_url=self.base_url.rstrip("/"),
                api_key=self.api_key
            )

    def build_prompt(self, question):
        message = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": question}
                ]
            }
        ]
        return message

    def call_gpt_eval(self, message, model_name, retries=10, wait_time=1, temperature=0.0):
        for i in range(retries):
            try:
                result = self.client.chat.completions.create(
                    model=model_name,
                    messages=message,
                    temperature=temperature
                )
                return result.choices[0].message.content
            except Exception as e:
                if i < retries - 1:
                    print(f"⚠️ API call failed ({i+1}/{retries}), retrying in {wait_time}s...")
                    print(e)
                    time.sleep(wait_time)
                else:
                    print("❌ Failed after all retries.")
                    raise e

    def inference(self, prompt, temperature=0.7):
        messages = self.build_prompt(prompt)
        response = self.call_gpt_eval(messages, self.model_name, temperature=temperature)
        return response


if __name__ == "__main__":
    # === Example: Azure OpenAI ===
    model = "gpt-4o"  # Azure의 '배포 이름(Deployment name)'
    base_url = "https://your-resource-name.openai.azure.com/"
    api_key = "your-azure-api-key"
    
    gpt = GPT(model, base_url, api_key)
    print(gpt.inference("Hello from Azure!"))

    # === Example: OpenAI ===
    # model = "gpt-4o"
    # base_url = "https://api.openai.com/v1"
    # api_key = "sk-..."
    # gpt = GPT(model, base_url, api_key)
    # print(gpt.inference("Hello from OpenAI!"))