# this file is responsible for talking with the LLM 
import requests
from langchain_core.messages import BaseMessage
import os
import json

OLLAMA_URL = os.getenv("OLLAMA_URL")
LLM_MODEL = os.getenv("LLM_MODEL")

def _to_ollama_messages(messages: list[BaseMessage]) -> list[dict]:
    LC_TO_OLLAMA_ROLE = {
        "human": "user",
        "ai": "assistant",
        "system": "system"
    }
    converted_msgs = []
    converted_msgs.append({"role": "system", "content": "You are a helpful assistant. Keep your responses short and concise."})
    for message in messages:
        converted_msgs.append({"role": LC_TO_OLLAMA_ROLE[message.type], "content": message.content})
    return converted_msgs

def call_LLM(messages: list[BaseMessage], timeout: int = 300) -> str:

    ollama_messages = _to_ollama_messages(messages)
    try:
            with requests.post(
            OLLAMA_URL,
            json={
                "model": LLM_MODEL,
                "messages": ollama_messages,
                "stream": True,
                "options": {
                "temperature": 0 
               }
            },
            timeout=timeout
            ) as response:
                
                # I will raise error if the status code is not in 200s 
                response.raise_for_status()
                # data = response.json()

                # # print("DATA FROM LLM:", data)

                # return data.get("message", {})

                for line in response.iter_lines():
                    if not line:
                        continue
                    
                    try:
                        data = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    
                    content = data.get("message", {}).get("content", "")

                    if content:
                        yield content
                    
                    if data.get("done"):
                        break
    
    except requests.exceptions.Timeout:
        print("Error: LLM request timed out ....")
        yield "Error: LLM request timed out ...."
    
    except requests.exceptions.ConnectionError:
        print("Error: Could not connect to Ollama ....")
        yield "Error: Could not connect to Ollama. Is it running?"

    except Exception as e:
        print(f"Error: {str(e)}")
        yield f"Error: {str(e)}"