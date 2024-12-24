## rag-ecommerce
Using a RAG agent to retrieve items from an ecommerce website with more ease than searching through the website.

# Scraping
In scrape.py I include the code used to scrape the END Clothing (https://www.endclothing.com/gb) website for all the items; including the name, brand, price, sizes available, descriptions, and image urls into a csv.

# Enriching Data
As an input to the vector store in the RAG agent, I wanted detailed descriptions of the items of clothing in natural language form that could then be turned into vectors using Nomic Embeddings and in that way get differentiation between items. In order to get the most rich and detailed descriptions of the items, I took the descriptions straight from the website, the bullet point description from the website as well a description using image analysis of the picture of the item, then combined this together into a final description which would then be inputted into the Vector Store. This is done in the categorize_images and get_full_description files.

# RAG
In rag_findAI I have written a basic RAG agent using langchain to collect the items based on an input query to ollama for what the user is looking for. 

# Next Steps
Currently it gets the right category of clothing and somewhat close to what the query is. But using it for myself, it is far from the powerful finder (fashion assistant) I had in mind. One of the issues is that the description of the items that I made, suffer a bit from hallucinations in the LLM, extra detail that is not relevant to the objective of the application (like descriptive language and metaphors lol); this is likely because my prompt asked for a "detailed description". 

I see 2 ways to improve this, 1 is to ask for a simplified description using more strict prompting, or I could also go in a different direction and instead of asking for a description in "prose", come up with a lot of key aspects of an item of clothing, e.g., colour, fit, luxury, ... and so on and create a detailed clothes feature dictionary and get an LLM to use the description and images of the item to fill out this dictionary; then simply creating a vector from this dictionary and use this in the vectorstore for the items.
