from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage
from openai import OpenAI

client = OpenAI(api_key="")
system_message = "You are acting as a helpful assistant, with a thick australian accent. You always end your messages with 'Mate' and your answers are always concise and less than two sentences"
model = "gpt-4o-mini"

def your_search_function(user_input: str):
    result = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": system_message
                },
                {
                    "role": "user",
                    "content": user_input
                }
            ]
        )
    
    return result.choices[0].message.content

if __name__ == "__main__":

    system_message = "You are acting as a helpful assistant, with a thick australian accent. You always end your messages with 'Mate' and your answers are always concise and less than two sentences"

    user_input = "What are the best places to visit in London"

    client = OpenAI(api_key=api_key)

    model = "gpt-4o-mini"

    result = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": system_message
                },
                {
                    "role": "user",
                    "content": user_input
                }
            ]
        )
    
    print(result.choices[0].message.content)
    