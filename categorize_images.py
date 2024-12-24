import time
import anthropic
import base64
import httpx
import ollama

start = time.time()

image_media_type = "image/jpeg"
model="claude-3-5-sonnet-20241022"
max_tokens=1024
image_url = "https://images.pexels.com/photos/1108099/pexels-photo-1108099.jpeg"

image_data = base64.standard_b64encode(httpx.get(image_url).content).decode("utf-8")

image_dict = {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": image_media_type,
                "data": image_data,
            },
        }

text_message = {
                        "type": "text",
                        "text": "What is in this image?"
                    }

content = [image_dict, text_message]

client = anthropic.Anthropic()

message = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        messages=[
            {
                "role": "user",
                "content": content,
            }
        ],
    )

print(message.content[0].text)

end = time.time()

elapsed_time = end - start
print(elapsed_time)

start = time.time()

response = ollama.chat(
    model='llama3.2-vision',
    messages=[{
        'role': 'user',
        'content': 'What is in this image?',
        'images': ['dogs.jpeg']
    }]
)

print(response)

end = time.time()

elapsed_time = end - start
print(elapsed_time)