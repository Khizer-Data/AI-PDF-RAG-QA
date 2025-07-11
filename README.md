# AI PDF RAG Q&A (Groq-Powered)

This project is a **Retrieval-Augmented Generation (RAG)** system for question answering over PDF documents. Users can upload PDFs, ask questions about their content, and receive answers generated by LLaMA 3 (via Groq Cloud), with relevant document snippets retrieved using semantic search.

## Features

- 📄 **PDF Upload:** Upload one or more PDF files.
- 🔍 **Semantic Search:** Extracts and chunks text, encodes with Sentence Transformers, and retrieves relevant content using FAISS vector search.
- 🤖 **LLM-Powered Answers:** Uses LLaMA 3 (Groq API) to generate answers based on retrieved context.
- 🏷️ **Domain Detection:** Automatically detects the document domain (Legal, Resume, Policy, etc.).
- 🖥️ **Streamlit UI:** Simple, interactive web interface.
- 📝 **Logging:** Logs key events and errors for debugging.

---

## Quickstart

### 1. Clone the Repository

```bash
git clone <your-repo-url>
cd rag-qa-legal-pdfs/app
```

### 2. Install Dependencies

It’s recommended to use a virtual environment.

```bash
pip install -r requirements.txt
```

**Required packages include:**
- streamlit
- sentence-transformers
- faiss-cpu
- pymupdf
- openai
- python-dotenv

### 3. Set Up Environment Variables

Create a `.env` file in the `app` directory with your Groq API key:

```
GROQ_API_KEY=your_groq_api_key_here
```

### 4. Run the App

```bash
streamlit run main.py
```

---

## How It Works

1. **Upload PDFs:** The app extracts text from each page of your PDFs.
2. **Chunking:** Text is split into overlapping chunks for better retrieval.
3. **Embedding:** Each chunk is converted to a vector using `all-MiniLM-L6-v2` from Sentence Transformers.
4. **Indexing:** All vectors are stored in a FAISS index for fast similarity search.
5. **Ask Questions:** Your question is embedded and compared to all chunks using cosine similarity.
6. **Context Retrieval:** The top-k most relevant chunks are selected.
7. **LLM Answering:** The context and question are sent to LLaMA 3 (via Groq) to generate an answer.

---

## File Structure

```
app/
├── main.py            # Streamlit app
├── rag_pipeline.py    # Core RAG pipeline (PDF parsing, chunking, embedding, retrieval)
├── requirements.txt   # Python dependencies
└── .env               # (You create this) API keys and secrets
```

---

## Configuration

- **Chunk Size & Overlap:** Adjust in `rag_pipeline.py` (`chunk_text` method).
- **LLM Model:** Change model name in `main.py` if needed.
- **Top-k Retrieval:** Adjustable via Streamlit slider.

---

## Troubleshooting

- **FAISS or PyMuPDF errors:** Ensure all dependencies are installed for your Python version.
- **Groq API errors:** Check your API key and network connection.
- **Large PDFs:** May take longer to process and index.

---

## Credits

- [Sentence Transformers](https://www.sbert.net/)
- [FAISS](https://github.com/facebookresearch/faiss)
- [PyMuPDF](https://github.com/pymupdf/PyMuPDF)
- [Streamlit](https://streamlit.io/)
- [Groq Cloud](https://console.groq.com/)

---

## License

MIT License

---

## Author

Muhammad Khizer Zakir
