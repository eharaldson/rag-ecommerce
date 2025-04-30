import pandas as pd 
import numpy as np 
import torch
import clip
from PIL import Image
from io import BytesIO
import requests
from multiprocessing import Pool, cpu_count
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError

# Set up device and load the CLIP model and its image preprocessing pipeline.
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# Load the CSV file
df = pd.read_csv('ProductStructuredInfo.csv', index_col=0)

df_len = len(df)
row_indx = 1

def get_image_embedding(image_paths):
    """
    Given a list of image paths for an item, this function:
    - Loads and preprocesses each image using CLIP's preprocessing.
    - Uses CLIP's image encoder to produce an embedding for each image.
    - Aggregates the embeddings (here, by averaging) to produce a single embedding.
    - Normalizes the resulting embedding so its length is 1.
    """
    embeddings = []
    for path in image_paths:
        # Open the image and convert to RGB in case it's grayscale or another format.
        response = requests.get(path)
        image = Image.open(BytesIO(response.content)).convert("RGB")
        # Preprocess the image (resize, crop, normalize, etc.) and add a batch dimension.
        image_input = preprocess(image).unsqueeze(0).to(device)
        with torch.no_grad():
            # Encode the image to get its embedding.
            image_embedding = model.encode_image(image_input)
        embeddings.append(image_embedding)
    
    # Concatenate embeddings along the batch dimension.
    embeddings = torch.cat(embeddings, dim=0)
    # Aggregate embeddings by taking the mean across all images.
    aggregated_embedding = embeddings.mean(dim=0, keepdim=True)
    # Normalize the aggregated embedding.
    aggregated_embedding = aggregated_embedding / aggregated_embedding.norm(dim=-1, keepdim=True)
    
    return aggregated_embedding

def get_image_embedding_for_item(row):
    """
    Given a row of the DataFrame, this function retrieves the image URLs,
    computes the image embeddings, and returns the aggregated embedding.
    """
    image_urls = [get_image_urls(row)[0]]
    if not image_urls:
        return torch.zeros(1, 512).to(device)
    return get_image_embedding(image_urls)

def get_image_urls(row):
    image_urls = row['image_urls']
    image_urls = [url for url in image_urls.split("'") if "http" in url]
    return image_urls

def get_image_embedding_with_timeout(image_urls, timeout=10):
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(get_image_embedding, image_urls)
        return future.result(timeout=timeout)
    
# your “lambda” – turn it into a named function
def process_row(row_dict):

    global row_indx 
    row_indx += 1

    print("\r"+str(row_indx)+" / "+str(df_len), end="", flush=True)

    # row_dict is a plain dict of your columns
    # e.g. return your_lambda(row_dict)
    image_urls = get_image_urls(row_dict)
    if not image_urls:
        return torch.zeros(1, 512).to(device).reshape(-1)
    else:
        try:
            res = get_image_embedding_with_timeout(image_urls)
            return res
        except TimeoutError:
            return torch.zeros(1, 512).to(device).reshape(-1)
    
if __name__ == "__main__":

    rows = df.to_dict('records')

    # 2) spin up a pool
    with Pool(processes=cpu_count()) as pool:
        # 3) map your function onto all rows
        results = pool.map(process_row, rows)

    stacked = torch.stack(results, dim=0).cpu().numpy()

    # 2) Save as CSV of floats
    #    fmt controls precision; here we show 6 decimal places
    np.savetxt("ImageEmbeddings.csv", stacked, delimiter=",", fmt="%.8f")