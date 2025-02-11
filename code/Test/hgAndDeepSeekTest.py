from huggingface_hub import InferenceClient
from dotenv import load_dotenv
import os

load_dotenv()



client = InferenceClient(
	provider="together",
	api_key=os.getenv("HF_TOKEN")
)

messages = [
	{
		"role": "user",
		"content": "What is the capital of France? just give me one word answer nothing else"
	}
]

completion = client.chat.completions.create(
    model="deepseek-ai/DeepSeek-R1", 
	messages=messages, 
	max_tokens=500
)

print(completion.choices[0].message)