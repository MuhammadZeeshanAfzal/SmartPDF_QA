import os
import fitz  # PyMuPDF for PDF text extraction
import faiss
import numpy as np
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
from sentence_transformers import SentenceTransformer
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index import SimpleDirectoryReader, VectorStoreIndex, ServiceContext
from llama_index.node_parser import SimpleNodeParser

# Initialize Flask app
app = Flask(__name__)

# Configurations
UPLOAD_FOLDER = "uploads"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = "mistralai/Mistral-7B-Instruct"

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load Embedding Model
embed_model = SentenceTransformer(EMBEDDING_MODEL)

# Load LLM
llm = HuggingFaceLLM(model_name=LLM_MODEL)

# FAISS Index Setup
dimension = embed_model.get_sentence_embedding_dimension()
index = faiss.IndexFlatL2(dimension)

# Dictionary to store document texts
doc_texts = {}


def extract_text_from_pdf(pdf_path):
    """Extract text from a PDF file."""
    text = ""
    with fitz.open(pdf_path) as doc:
        for page in doc:
            text += page.get_text("text") + "\n"
    return text


def index_document(doc_id, text):
    """Chunk text, generate embeddings, and index in FAISS."""
    global index
    parser = SimpleNodeParser.from_defaults()
    nodes = parser.get_nodes_from_documents([text])
    
    # Store document texts
    doc_texts[doc_id] = text

    embeddings = np.array([embed_model.encode(node.text) for node in nodes], dtype=np.float32)
    index.add(embeddings)


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

    # Process and index the document
    text = extract_text_from_pdf(file_path)
    index_document(filename, text)

    return jsonify({"message": "File uploaded and indexed successfully"})


@app.route("/query", methods=["POST"])
def query():
    """Handle user queries and return answers."""
    data = request.json
    user_query = data.get("query")

    if not user_query:
        return jsonify({"error": "Query is required"}), 400

    # Convert query to embedding
    query_embedding = embed_model.encode(user_query).reshape(1, -1)

    # Search in FAISS
    D, I = index.search(query_embedding, k=3)
    results = [doc_texts.get(idx, "No match found") for idx in I[0]]

    # Generate answer using LLM
    context = " ".join(results)
    prompt = f"Based on the following context, answer the question:\n\nContext: {context}\n\nQuestion: {user_query}\n\nAnswer:"
    response = llm.complete(prompt)

    return jsonify({"answer": response})


if __name__ == "__main__":
    app.run(debug=True)
