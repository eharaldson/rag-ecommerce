from flask import Flask, request, jsonify, render_template, session
import os
from werkzeug.utils import secure_filename
from rag_backend import RAGSearchEngine
import base64
import tempfile
import secrets

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['UPLOAD_FOLDER'] = tempfile.gettempdir()
app.secret_key = secrets.token_hex(16)  # For session management

# Initialize the RAG search engine
search_engine = RAGSearchEngine()

# Route to serve the HTML page
@app.route("/")
def home():
    return render_template("website.html")

# API endpoint to perform the search
@app.route("/search", methods=["POST"])
def search():
    try:
        data = request.get_json()
        
        # Initialize conversation history in session if not exists
        if 'conversation_history' not in session:
            session['conversation_history'] = []
        
        # Get conversation history
        conversation_history = session['conversation_history']
        
        # Handle text query
        if 'query' in data and data['query']:
            query_text = data['query']
            results = search_engine.search_by_text(query_text, conversation_history)
            
            # Update conversation history
            conversation_history.append({
                'role': 'user',
                'content': query_text
            })
            conversation_history.append({
                'role': 'assistant',
                'content': results['message']
            })
            
        # Handle image query (base64 encoded)
        elif 'image' in data and data['image']:
            # Decode base64 image
            image_data = data['image'].split(',')[1] if ',' in data['image'] else data['image']
            
            # Save temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
                tmp_file.write(base64.b64decode(image_data))
                tmp_path = tmp_file.name
            
            try:
                results = search_engine.search_by_image(tmp_path, conversation_history)
                
                # Update conversation history
                conversation_history.append({
                    'role': 'user',
                    'content': '[User uploaded an image]'
                })
                conversation_history.append({
                    'role': 'assistant',
                    'content': results['message']
                })
                
            finally:
                # Clean up temp file
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
        else:
            return jsonify({"error": "No query or image provided"}), 400
        
        # Keep only last 10 exchanges (20 messages) to prevent session bloat
        if len(conversation_history) > 20:
            conversation_history = conversation_history[-20:]
        
        # Save updated history to session
        session['conversation_history'] = conversation_history
        session.modified = True
            
        return jsonify(results)
        
    except Exception as e:
        print(f"Error in search endpoint: {str(e)}")
        return jsonify({"error": str(e)}), 500

# Endpoint to clear conversation history
@app.route("/clear_history", methods=["POST"])
def clear_history():
    session.pop('conversation_history', None)
    return jsonify({"status": "success", "message": "Conversation history cleared"})

# Health check endpoint
@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "healthy"}), 200

if __name__ == "__main__":
    app.run(debug=True, port=5000)