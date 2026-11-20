import os
from openai import openai
from dotenv import load_dotenv
from tool import TOOLS

load_dotenv()

client = openai(
    base_url = os.getenv("BASE_URL"),
    api_key = os.getenv("API_KEY")
)

stream = client.chat.completions.create(
    model=os.getenv("MODEL"),
    messages =[{"role": "user" ,"content":"Tell me weather of mumbai and aligarh"}] ,
    tools = TOOLS
    stream = True,
)

calls = {}

for chunk in stream:
    if not chunk.choices:
        continue
    
    delta = chunk.choices[0].delta

    for tc in (delta.tool_calls or []):
        obj = calls.setdefault(
            tc.index,
            SimpleNamespace(id = None , fucntion=SimpleNamespace(name="" , arguments=""))
        )
        if tc.id:
            obj.id = tc.id
        if tc.function and tc.function.name:
            obj.function.name += tc.function.name
        if tc.function and tc.function.arguments:
            obj.function.arguments += tc.function.arguments
        
call_list = []

for key in sorted(calls):
    call_list.append(calls(key))
    