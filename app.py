import os
import fitz  # PyMuPDF for PDF text extraction
import faiss
import numpy as np
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.huggingface import HuggingFaceInferenceAPI
from llama_index import Document, SimpleDirectoryReader, VectorStoreIndex, ServiceContext
from llama_index.node_parser import SimpleNodeParser

# Initialize Flask app
app = Flask(__name__)

# Configurations
UPLOAD_FOLDER = "uploads"
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
LLM_MODEL = "mistralai/Mistral-7B-Instruct-v0.3"
FAISS_INDEX_PATH = "faiss_index.index"

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load Embedding Model
embed_model = HuggingFaceEmbedding(model_name=EMBEDDING_MODEL)

# Load LLM
llm = HuggingFaceInferenceAPI(model_name=LLM_MODEL, token=os.getenv("HUGGINGFACE_TOKEN"))

# FAISS Index Setup
dimension = embed_model.get_sentence_embedding_dimension()
if os.path.exists(FAISS_INDEX_PATH):
    index = faiss.read_index(FAISS_INDEX_PATH)
else:
    index = faiss.IndexFlatL2(dimension)

# Save the index before the application exits
import atexit
atexit.register(lambda: faiss.write_index(index, FAISS_INDEX_PATH))

# Dictionary to store document texts and their corresponding FAISS indices
doc_texts = {}
doc_id_to_faiss_idx = {}
current_faiss_idx = 0


def extract_text_from_pdf(pdf_path):
    """Extract text from a PDF file."""
    text = ""
    with fitz.open(pdf_path) as doc:
        for page in doc:
            text += page.get_text("text") + "\n"
    return text


def index_document(doc_id, text):
    """Chunk text, generate embeddings, and index in FAISS."""
    global index, current_faiss_idx
    parser = SimpleNodeParser.from_defaults()
    nodes = parser.get_nodes_from_documents([Document(text=text)])
    
    # Store document texts
    doc_texts[doc_id] = text

    # Generate embeddings and add to FAISS index
    embeddings = np.array([embed_model.encode(node.text) for node in nodes], dtype=np.float32)
    index.add(embeddings)
    
    # Map FAISS indices to document IDs
    for _ in range(len(nodes)):
        doc_id_to_faiss_idx[current_faiss_idx] = doc_id
        current_faiss_idx += 1


@app.route("/upload", methods=["POST"])
def upload_pdf():
    """Handle PDF uploads and indexing."""
    if "file" not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(file_path)

    try:
        # Process and index the document
        text = extract_text_from_pdf(file_path)
        index_document(filename, text)
    except Exception as e:
        return jsonify({"error": f"Failed to process the file: {str(e)}"}), 500

    return jsonify({"message": "File uploaded and indexed successfully"})


@app.route("/query", methods=["POST"])
def query():
    """Handle user queries and return answers."""
    data = request.json
    user_query = data.get("query")

    if not user_query:
        return jsonify({"error": "Query is required"}), 400

    try:
        # Convert query to embedding
        query_embedding = embed_model.encode(user_query).reshape(1, -1)

        # Search in FAISS
        D, I = index.search(query_embedding, k=3)
        results = [doc_texts[doc_id_to_faiss_idx[idx]] for idx in I[0]]

        # Generate answer using LLM
        context = " ".join(results)
        prompt = f"Based on the following context, answer the question:\n\nContext: {context}\n\nQuestion: {user_query}\n\nAnswer:"
        response = llm.complete(prompt)

        # Clean the response
        cleaned_response = response.strip()

        return jsonify({"answer": cleaned_response})
    except Exception as e:
        return jsonify({"error": f"Failed to process the query: {str(e)}"}), 500


if __name__ == "__main__":
    app.run(debug=os.getenv("FLASK_DEBUG", "False").lower() == "true")