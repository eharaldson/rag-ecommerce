from flask import Flask, request, jsonify, render_template
from findAItest import your_search_function

app = Flask(__name__)

# Route to serve a simple HTML page (for a UI)
@app.route("/")
def home():
    return render_template("website.html")  # Create this template to provide an input form

# API endpoint to perform the search
@app.route("/search", methods=["POST"])
def search():
    # Expect JSON data with a 'query' field
    data = request.get_json()
    query = data.get("query")
    
    # Call your existing Python code to search the database using the LLM API
    results = your_search_function(query)  # Replace with your actual function
    
    return jsonify(results=results)

if __name__ == "__main__":
    app.run(debug=True)
