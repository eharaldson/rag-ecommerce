import pandas as pd 
import numpy as np 
import torch
import clip
from multiprocessing import Pool, cpu_count

structured_data_for_embeddings = ['fit', 'specific_category', 'primary_colour', 'secondary_colour', 'pattern', 'fabric', 'price_category']

def get_structured_encoding_texts(row: pd.Series):

    structured_encoding_texts = []

    for column in structured_data_for_embeddings:
        if type(row[column]) == str:
            structured_encoding_texts.append(row[column])

    return structured_encoding_texts

def get_descriptive_encoding_texts(row: pd.Series):

    descriptive_encoding_texts = []

    descriptive_encoding_texts += [x for x in row['additional_features'].split(',') if type(x) == str]
    descriptive_encoding_texts += row['description']

    return descriptive_encoding_texts

def get_key_word_encoding_texts(row: pd.Series):

    return [x for x in row['key_words'].split(',') if type(x) == str]

device = "cpu"
i = 0

# Load the pre-trained CLIP model and its tokenizer.
model, preprocess = clip.load("ViT-B/32", device=device)

def get_descriptive_embeddings(row: dict):

    descriptive_text = get_structured_encoding_texts(row)
    descriptive_text_tokens = clip.tokenize(descriptive_text).to(device)

    # Generate embeddings for each descriptor.
    with torch.no_grad():
        descriptive_text_embeddings = model.encode_text(descriptive_text_tokens)

    item_descriptive_info_embedding = descriptive_text_embeddings.mean(dim=0, keepdim=True)

    item_descriptive_info_embedding = item_descriptive_info_embedding / item_descriptive_info_embedding.norm(dim=-1, keepdim=True)

    return item_descriptive_info_embedding.reshape(-1)

def get_key_word_embeddings(row: dict):

    key_word_text = get_structured_encoding_texts(row)
    key_word_text_tokens = clip.tokenize(key_word_text).to(device)

    # Generate embeddings for each descriptor.
    with torch.no_grad():
        key_word_text_embeddings = model.encode_text(key_word_text_tokens)

    item_key_word_info_embedding = key_word_text_embeddings.mean(dim=0, keepdim=True)

    item_key_word_info_embedding = item_key_word_info_embedding / item_key_word_info_embedding.norm(dim=-1, keepdim=True)

    return item_key_word_info_embedding.reshape(-1)

def get_structured_embeddings(row: dict):

    structured_text = get_structured_encoding_texts(row)
    structured_text_tokens = clip.tokenize(structured_text).to(device)

    # Generate embeddings for each descriptor.
    with torch.no_grad():
        structured_text_embeddings = model.encode_text(structured_text_tokens)

    item_structured_info_embedding = structured_text_embeddings.mean(dim=0, keepdim=True)
    item_structured_info_embedding = item_structured_info_embedding / item_structured_info_embedding.norm(dim=-1, keepdim=True)

    return item_structured_info_embedding.reshape(-1)

df = pd.read_csv('ProductStructuredInfo.csv', index_col=0)
df_len = len(df)
row_indx = 1

def process_row_descriptive(row_dict: dict):

    global row_indx 
    row_indx += 1

    print("\r"+str(row_indx)+" / "+str(df_len), end="", flush=True)

    return get_descriptive_embeddings(row_dict)

def process_row_key_word(row_dict: dict):

    global row_indx 
    row_indx += 1

    print("\r"+str(row_indx)+" / "+str(df_len), end="", flush=True)

    return get_key_word_embeddings(row_dict)

def process_row_structured(row_dict: dict):

    global row_indx 
    row_indx += 1

    print("\r"+str(row_indx)+" / "+str(df_len), end="", flush=True)

    return get_structured_embeddings(row_dict)

if __name__ == "__main__":

    rows = df.to_dict('records')

    # Descriptive Embeddings
    with Pool(processes=cpu_count()) as pool:
        # 3) map your function onto all rows
        results = pool.map(process_row_descriptive, rows)

    stacked = torch.stack(results, dim=0).cpu().numpy()

    # 2) Save as CSV of floats
    #    fmt controls precision; here we show 6 decimal places
    np.savetxt("DescriptiveEmbeddings.csv", stacked, delimiter=",", fmt="%.8f")

    # Keyword Embeddings
    with Pool(processes=cpu_count()) as pool:
        # 3) map your function onto all rows
        results = pool.map(process_row_key_word, rows)

    stacked = torch.stack(results, dim=0).cpu().numpy()

    # 2) Save as CSV of floats
    #    fmt controls precision; here we show 6 decimal places
    np.savetxt("KeywordEmbeddings.csv", stacked, delimiter=",", fmt="%.8f")

    # Structured Embeddings
    with Pool(processes=cpu_count()) as pool:
        # 3) map your function onto all rows
        results = pool.map(process_row_structured, rows)

    stacked = torch.stack(results, dim=0).cpu().numpy()

    # 2) Save as CSV of floats
    #    fmt controls precision; here we show 6 decimal places
    np.savetxt("StructuredEmbeddings.csv", stacked, delimiter=",", fmt="%.8f")