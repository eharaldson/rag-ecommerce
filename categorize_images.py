import time
import anthropic
import base64
import httpx
import ollama
import requests

def get_base64_image_from_url(url):
    response = requests.get(url)
    response.raise_for_status()  # Raise an error if the request fails
    return base64.b64encode(response.content).decode('utf-8')

image_url = "https://images.pexels.com/photos/1108099/pexels-photo-1108099.jpeg"
encoded_image = get_base64_image_from_url(image_url)

start = time.time()

response = ollama.chat(
    model='llama3.2-vision',
    messages=[{
        'role': 'user',
        'content': 'What is in this image?',
        'images': [encoded_image]
    }]
)

print(response)

end = time.time()

elapsed_time = end - start
print(elapsed_time)