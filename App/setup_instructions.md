# FindAI - Fashion Search Application

A Flask-based application that uses CLIP embeddings and RAG (Retrieval-Augmented Generation) to search for fashion items using text or image queries.

## Features

- **Text Search**: Describe what you're looking for in natural language
- **Image Search**: Upload an image to find similar items
- **AI-Powered Responses**: Get conversational responses about found items
- **Visual Results**: See product images, prices, and descriptions in a side panel

## Setup Instructions

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Install CLIP

```bash
pip install git+https://github.com/openai/CLIP.git
```

### 3. Set Up Environment Variables

Create a `.env` file in the project root:

```bash
# For OpenAI (if using GPT models)
OPENAI_API_KEY=your_api_key_here

# For Tavily search (if needed)
TAVILY_API_KEY=your_tavily_key_here
```

### 4. Prepare Your Data

Ensure you have the following CSV files in your project directory:
- `ProductStructuredInfo.csv` - Main product data
- `ProductStructuredInfoWithEmbeddings.csv` (optional) - Pre-computed embeddings
- Or use the embedding scripts to generate embeddings:
  ```bash
  python embed_text_data.py
  python embed_image_data.py
  ```

### 5. Configure the Backend

In `rag_backend.py`, you can configure:
- `use_openai`: Set to `True` for OpenAI GPT models or `False` for local Ollama
- Adjust the number of search results returned
- Modify the LLM prompts for different response styles

### 6. Run the Application

```bash
python app.py
```

The application will be available at `http://localhost:5000`

## Usage

1. **Text Search**: 
   - Type a description like "black hoodie from Gucci or YSL, oversized"
   - Click "Send" to search

2. **Image Search**:
   - Click "Upload Image" and select a fashion item image
   - The system will find similar items

3. **View Results**:
   - Chat responses appear in the left panel
   - Product cards with images and details appear in the right panel

## Project Structure

```
findai/
├── app.py                  # Flask application
├── rag_backend.py         # RAG search engine implementation
├── templates/
│   └── website.html       # Frontend interface
├── embed_text_data.py     # Text embedding generation
├── embed_image_data.py    # Image embedding generation
├── findAItest.py          # Test functions
├── requirements.txt       # Python dependencies
└── data/
    ├── ProductStructuredInfo.csv
    └── ProductStructuredInfoWithEmbeddings.csv
```

## Customization

### Adding New Categories
Edit the `categories` list in `rag_backend.py`

### Changing the LLM
- For OpenAI: Set `use_openai = True` and configure your API key
- For Ollama: Set `use_openai = False` and ensure Ollama is running locally

### Modifying the UI
Edit `templates/website.html` to change colors, layout, or add new features

### Improving Search Quality
- Adjust the `k` parameter in the retriever for more/fewer results
- Modify the embedding generation prompts
- Fine-tune the similarity thresholds

## Troubleshooting

### Common Issues

1. **Missing CSV files**: Ensure your product data CSV files are in the project root directory

2. **CUDA/GPU errors**: The system will automatically fall back to CPU if CUDA is not available

3. **API key errors**: Make sure your OpenAI API key is set correctly in the environment

4. **Memory issues**: Reduce batch size in embedding generation scripts or process data in chunks

5. **Port already in use**: Change the port in `app.py` or kill the process using port 5000

### Performance Optimization

- Pre-compute and save embeddings to avoid regenerating them
- Use GPU acceleration when available
- Implement caching for frequently searched queries
- Consider using a proper database instead of CSV files for larger datasets

## Future Enhancements

- Add user authentication and saved searches
- Implement filtering by price, brand, category
- Add shopping cart functionality
- Integrate with e-commerce APIs
- Improve image similarity search with pre-computed CLIP embeddings
- Add multi-language support

## License

This project is for educational and demonstration purposes.