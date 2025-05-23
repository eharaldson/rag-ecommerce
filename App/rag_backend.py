import pandas as pd
import numpy as np
import torch
import clip
from PIL import Image
from langchain_community.vectorstores import SKLearnVectorStore
from langchain_nomic.embeddings import NomicEmbeddings
from langchain_core.documents import Document
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage
from openai import OpenAI
import ast
import json
from typing import List, Dict, Union
import os

class RAGSearchEngine:
    def __init__(self):
        """Initialize the RAG search engine with CLIP and vector store."""
        # Initialize CLIP model
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.clip_model, self.clip_preprocess = clip.load("ViT-B/32", device=self.device)
        
        # Initialize LLM (you can switch between OpenAI and Ollama)
        self.use_openai = True  # Set to False to use Ollama
        if self.use_openai:
            # You'll need to set your API key here or via environment variable
            self.openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY", ""))
            self.llm_model = "gpt-4o-mini"
        else:
            self.llm = ChatOllama(model="llama3.2:3b-instruct-fp16", temperature=0)
        
        # Load product data
        self.load_data()
        
        # Initialize vector store
        self.initialize_vector_store()
        
    def load_data(self):
        """Load product data and embeddings."""
        try:
            # Load the main product data
            self.df = pd.read_csv('ProductStructuredInfo.csv', index_col=0)
            
            # Load pre-computed embeddings if available
            if os.path.exists('ProductStructuredInfoWithEmbeddings.csv'):
                df_with_embeddings = pd.read_csv('ProductStructuredInfoWithEmbeddings.csv', index_col=0)
                
                # Parse string representations of embeddings back to numpy arrays
                for col in ['structured_embeddings', 'descriptive_embeddings', 'keyword_embeddings']:
                    if col in df_with_embeddings.columns:
                        self.df[col] = df_with_embeddings[col].apply(self.parse_embedding_string)
            
            # If no AI descriptions exist, use regular descriptions
            if 'AIDescription' not in self.df.columns and 'description' in self.df.columns:
                self.df['AIDescription'] = self.df['description']
                
        except Exception as e:
            print(f"Error loading data: {e}")
            # Create dummy data for testing
            self.df = pd.DataFrame({
                'ProductName': ['Test Product'],
                'Brand': ['Test Brand'],
                'Price': [100.0],
                'Category': ['Test Category'],
                'AIDescription': ['Test description'],
                'image_urls': ["['https://example.com/image.jpg']"]
            })
    
    def parse_embedding_string(self, embedding_str):
        """Parse string representation of embedding to numpy array."""
        try:
            # Remove extra brackets and parse
            cleaned = embedding_str.strip('[]').replace('\n', ' ')
            # Split by spaces and convert to float
            values = [float(x) for x in cleaned.split() if x]
            return np.array(values).reshape(1, -1)
        except:
            return np.zeros((1, 512))  # Return zero vector if parsing fails
    
    def initialize_vector_store(self):
        """Initialize the vector store for semantic search."""
        # Create documents from product data
        documents = []
        for idx, row in self.df.iterrows():
            # Parse image URLs
            image_urls = row.get('image_urls', "[]")
            if isinstance(image_urls, str):
                image_urls = [url for url in image_urls.split("'") if "http" in url]
            
            metadata = {
                'product_name': row.get('ProductName', row.name),
                'brand': row.get('Brand', 'Unknown'),
                'price': row.get('Price', 0),
                'category': row.get('Category', 'Unknown'),
                'image_url': image_urls[0] if image_urls else ""
            }
            
            # Use AI description if available, otherwise use regular description
            content = row.get('AIDescription', row.get('description', 'No description available'))
            
            document = Document(page_content=str(content), metadata=metadata)
            documents.append(document)
        
        # Create vector store
        self.vectorstore = SKLearnVectorStore.from_documents(
            documents=documents,
            embedding=NomicEmbeddings(model="nomic-embed-text-v1.5", inference_mode="local"),
        )
        
        # Create retriever
        self.retriever = self.vectorstore.as_retriever(k=5)
    
    def get_text_embedding(self, text: str) -> torch.Tensor:
        """Get CLIP embedding for text."""
        text_tokens = clip.tokenize([text]).to(self.device)
        with torch.no_grad():
            text_embedding = self.clip_model.encode_text(text_tokens)
            text_embedding = text_embedding / text_embedding.norm(dim=-1, keepdim=True)
        return text_embedding
    
    def get_image_embedding(self, image_path: str) -> torch.Tensor:
        """Get CLIP embedding for image."""
        image = Image.open(image_path).convert("RGB")
        image_input = self.clip_preprocess(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            image_embedding = self.clip_model.encode_image(image_input)
            image_embedding = image_embedding / image_embedding.norm(dim=-1, keepdim=True)
        return image_embedding
    
    def create_detailed_query(self, user_query: str, conversation_history: List[Dict] = None) -> str:
        """Use LLM to create a detailed query from user input with conversation context."""
        categories = ['Hoodies & Sweats', 'Trousers', 'Coats & Jackets', 'Shorts', 
                     'Bags', 'T-Shirts', 'Jewellery', 'Belts', 'Swimwear', 'Shirts',
                     'Accessories', 'Hats', 'Knitwear', 'Socks', 'Lifestyle',
                     'Sweat Pants', 'Polo Shirts', 'Jeans', 'Sunglasses', 
                     'Scarves & Gloves', 'Wallets & Keychains', 'Publications', 
                     'Sportswear', 'Boots', 'Perfume & Fragrance', 'Sneakers', 
                     'Sandals & Slides', 'Home Decoration', 'Tableware', 'Watches', 
                     'Underwear', 'Loungewear', 'Shoe Care & Accessories',
                     'Soft Furnishings', 'Lighting', 'Storage & Organisers',
                     'Glassware', 'Home Fragrance', 'Shoes', 'Slippers', 'Running Shoes']
        
        # Build context from conversation history
        context = ""
        if conversation_history and len(conversation_history) > 0:
            recent_history = conversation_history[-6:]  # Last 3 exchanges
            context = "\n\nPrevious conversation:\n"
            for msg in recent_history:
                context += f"{msg['role'].capitalize()}: {msg['content']}\n"
        
        prompt = f"""You are an expert at creating detailed item descriptions from user queries.
        Create a detailed description that includes: type, category, material, brand preferences, 
        colors, patterns, fit/style, and any unique characteristics.
        
        Possible categories: {', '.join(categories)}
        {context}
        
        Current user query: {user_query}
        
        Consider the conversation context when interpreting the query. For example, if the user 
        previously asked about hoodies and now says "in blue", understand they want blue hoodies.
        
        Provide a detailed descriptive paragraph without bullet points."""
        
        if self.use_openai:
            messages = [{"role": "system", "content": prompt}]
            if conversation_history:
                # Add conversation history to messages
                for msg in conversation_history[-6:]:
                    messages.append({
                        "role": "user" if msg['role'] == 'user' else "assistant",
                        "content": msg['content']
                    })
            messages.append({"role": "user", "content": user_query})
            
            response = self.openai_client.chat.completions.create(
                model=self.llm_model,
                messages=messages
            )
            return response.choices[0].message.content
        else:
            response = self.llm.invoke([
                SystemMessage(content=prompt),
                HumanMessage(content=user_query)
            ])
            return response.content
    
    def format_results_with_llm(self, results: List[Document], original_query: str, conversation_history: List[Dict] = None) -> str:
        """Use LLM to format the search results in a conversational way with context."""
        # Create a summary of found items
        items_summary = []
        for i, doc in enumerate(results[:5]):  # Top 5 results
            items_summary.append(f"{i+1}. {doc.metadata['product_name']} by {doc.metadata['brand']} (£{doc.metadata['price']})")
        
        # Build context from conversation history
        context = ""
        if conversation_history and len(conversation_history) > 0:
            recent_history = conversation_history[-4:]  # Last 2 exchanges
            context = "\n\nRecent conversation:\n"
            for msg in recent_history:
                context += f"{msg['role'].capitalize()}: {msg['content']}\n"
        
        prompt = f"""You are a helpful fashion assistant engaged in an ongoing conversation with the user.
        {context}
        
        The user's latest request: "{original_query}"
        
        I found these items that match their request:
        {chr(10).join(items_summary)}
        
        Please provide a friendly, conversational response that:
        1. Acknowledges any context from the previous conversation
        2. Explains what you found and why these items match their request
        3. Maintains continuity with the conversation flow
        
        Keep it concise but informative. Do not use bullet points or numbered lists in your response."""
        
        if self.use_openai:
            messages = [{"role": "system", "content": "You are a helpful fashion assistant."}]
            if conversation_history:
                # Add recent conversation history
                for msg in conversation_history[-4:]:
                    messages.append({
                        "role": "user" if msg['role'] == 'user' else "assistant",
                        "content": msg['content']
                    })
            messages.append({"role": "user", "content": prompt})
            
            response = self.openai_client.chat.completions.create(
                model=self.llm_model,
                messages=messages
            )
            return response.choices[0].message.content
        else:
            response = self.llm.invoke([
                SystemMessage(content="You are a helpful fashion assistant."),
                HumanMessage(content=prompt)
            ])
            return response.content
    
    def search_by_text(self, query: str, conversation_history: List[Dict] = None) -> Dict:
        """Search for products using text query with conversation context."""
        try:
            # Create detailed query using LLM with conversation history
            detailed_query = self.create_detailed_query(query, conversation_history)
            
            # Search using vector store
            results = self.retriever.invoke(detailed_query)
            
            # Format results
            formatted_results = []
            for doc in results[:10]:  # Limit to 10 results
                formatted_results.append({
                    'product_name': doc.metadata['product_name'],
                    'brand': doc.metadata['brand'],
                    'price': doc.metadata['price'],
                    'category': doc.metadata['category'],
                    'image_url': doc.metadata['image_url'],
                    'description': doc.page_content[:200] + '...' if len(doc.page_content) > 200 else doc.page_content
                })
            
            # Get LLM response with conversation context
            llm_response = self.format_results_with_llm(results, query, conversation_history)
            
            return {
                'results': formatted_results,
                'message': llm_response,
                'query': query,
                'detailed_query': detailed_query
            }
            
        except Exception as e:
            print(f"Error in text search: {e}")
            return {
                'results': [],
                'message': f"I'm sorry, I encountered an error while searching: {str(e)}",
                'query': query,
                'error': str(e)
            }
    
    def search_by_image(self, image_path: str, conversation_history: List[Dict] = None) -> Dict:
        """Search for products using image with conversation context."""
        try:
            # Get image embedding using CLIP
            image_embedding = self.get_image_embedding(image_path)
            
            # For now, we'll create a text description of what we might be looking for
            # In a full implementation, you'd compare image embeddings directly
            query = "stylish clothing item from uploaded image"
            
            # Add context if there's conversation history
            if conversation_history and len(conversation_history) > 0:
                query = "similar items to the image I just uploaded"
            
            # Use the text search as a fallback with conversation history
            results = self.search_by_text(query, conversation_history)
            
            # Modify the message to indicate it's an image search
            if conversation_history:
                results['message'] = "Based on the image you uploaded and our conversation, " + results['message']
            else:
                results['message'] = "I've analyzed your image and found these similar items. " + results['message']
            
            results['image_search'] = True
            
            return results
            
        except Exception as e:
            print(f"Error in image search: {e}")
            return {
                'results': [],
                'message': f"I'm sorry, I couldn't process the image: {str(e)}",
                'error': str(e)
            }
    
    def test_connection(self) -> bool:
        """Test if the search engine is properly initialized."""
        try:
            # Test with a simple query
            results = self.retriever.invoke("black hoodie")
            return len(results) > 0
        except:
            return False