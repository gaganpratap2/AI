import os
import json
from openai import OpenAI
from dotenv import load_dotenv
from tool import TOOLS , run_tool
from utilities.utils import log


load_dotenv()


def main():

    print("Connecting to local Ollama...")

    client = OpenAI(
        base_url=os.getenv("BASE_URL"),
        api_key=os.getenv("API_KEY")
    )

    model = os.getenv("MODEL")

    messages = [
        {
            "role": "system",
            "content": "You are a helpful AI assistant. Keep answers under 50 words."
        }
    ]

    print(f"Connected to {model}!")
    print("Type 'exit', 'quit', or 'bye' to quit.")

    while True:

        query = input("\nUser: ").strip()

        if query.lower() in ["exit", "quit", "bye", "goodbye"]:
            print("Assistant: Goodbye!")
            return

        messages.append({
            "role": "user",
            "content": query
        })

        try:
            print("\nThinking...")

            

            for step in range(11):
                response = client.chat.completions.create(
                                model=model,
                                messages=messages,
                                tools=TOOLS,
                                stream=True
                )
                
                answer = response.choices[0].delta.content

                if not answer.tool_calls:
                    break

                log(f"Step {step + 1}: model requested {len(answer.tool_calls)}")
                messages.append(answer)

                for tool_call in answer.tool_calls:
                    result = run_tool(tool_call)
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": str(result)
                    })
            else:
                fallback = "Sorry , I could not finish this within the steps limits."
                messages.append({
                    "role":"assistant",
                    "content":fallback
                })
                print("Assistant: " + fallback)

            # Save assistant response
            messages.append({
                "role": "assistant",
                "content": answer.content or ""
            })

            print("\nAssistant:", answer.content)
            continue

        except Exception as e:
            print("\nError:", e)


if __name__ == "__main__":
    main()
