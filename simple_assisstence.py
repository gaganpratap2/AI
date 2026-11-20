
import os
import json
from types import SimpleNamespace

from openai import OpenAI
from dotenv import load_dotenv

from tool import TOOLS, run_tool
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
            "content": "You are a helpful AI assistant. Keep answers under 200 words."
        }
    ]

    print(f"Connected to {model}!")
    print("Type 'exit', 'quit', or 'bye' to quit.")

    def stream_chat(history):

        stream = client.chat.completions.create(
            model=model,
            messages=history,
            tools=TOOLS,
            stream=True
        )

        full_response = ""
        calls = {}
        label_printed = False

        for chunk in stream:

            if not chunk.choices:
                continue

            delta = chunk.choices[0].delta

            # Stream assistant text
            if delta.content:
                if not label_printed:
                    print("Assistant:", end=" ", flush=True)
                    label_printed = True

                print(delta.content, end="", flush=True)
                full_response += delta.content

            # Collect streamed tool calls
            for tc in (delta.tool_calls or []):

                obj = calls.setdefault(
                    tc.index,
                    SimpleNamespace(
                        id=None,
                        function=SimpleNamespace(
                            name="",
                            arguments=""
                        )
                    )
                )

                if tc.id:
                    obj.id = tc.id

                if tc.function:
                    if tc.function.name:
                        obj.function.name += tc.function.name

                    if tc.function.arguments:
                        obj.function.arguments += tc.function.arguments

        if label_printed:
            print()

        call_list = [
            calls[key]
            for key in sorted(calls)
        ]

        return full_response, call_list

    def build_tool_call_payload(tool_calls):

        payload = []

        for tool_call in tool_calls:

            tool_call_dict = {
                "id": tool_call.id,
                "type": "function",
                "function": {
                    "name": tool_call.function.name,
                    "arguments": tool_call.function.arguments
                }
            }

            payload.append(tool_call_dict)

        return payload

    # Main conversation loop
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

                full_response, tool_calls = stream_chat(messages)

                # No more tool calls: model has finished responding
                if not tool_calls:
                    messages.append({
                        "role": "assistant",
                        "content": full_response
                    })
                    break

                log(
                    f"Step {step + 1}: model requested "
                    f"{len(tool_calls)} tool(s)"
                )

                # Add assistant tool-call message
                messages.append({
                    "role": "assistant",
                    "content": full_response or None,
                    "tool_calls": build_tool_call_payload(tool_calls)
                })

                # Execute tools
                for tool_call in tool_calls:

                    try:
                        result = run_tool(tool_call)

                    except Exception as tool_error:
                        result = f"Tool execution error: {tool_error}"

                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": str(result)
                    })

            else:

                fallback = (
                    "Sorry, I could not finish within "
                    "the step limit."
                )

                messages.append({
                    "role": "assistant",
                    "content": fallback
                })

                print("Assistant:", fallback)

        except Exception as e:
            print("\nError:", e)


if __name__ == "__main__":
    main()
