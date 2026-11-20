import os
from openai import openai
from dotenv import load_dotenv

load_dotenv()

client = openai(
    base_url = os.getenv("BASE_URL"),
    api_key = os.getenv("API_KEY")
)

stream = client.chat.completions.create(
    model=os.getenv("MODEL"),
    messages =[{"role": "user" ,"content":"Explain what is api in under 100 words"}] ,
    stream = True,
)

full_response = ""

for chunk in stream:
    if not chunk.choices:
        continue
    
    piece = chunk.choices[0].delta.content
    if piece:
        print(piece , end="", flush=True)
        full_response += piece


print("\nfull Response")
print(full_response)