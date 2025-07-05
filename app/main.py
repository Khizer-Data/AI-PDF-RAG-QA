# main.py

import os
import streamlit as st
import tempfile
from rag_pipeline import RAGPipeline
from openai import OpenAI
import time
from dotenv import load_dotenv
import logging

# Set up logging to file
logging.basicConfig(
    filename="app.log",
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s"
)

load_dotenv()

client = OpenAI(
    api_key=os.getenv("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1"
)

st.set_page_config(page_title="AI PDF RAG (Groq-Powered)", layout="wide", page_icon="📚")
st.title("AI PDF RAG Q&A with Groq 🧠")
st.markdown("Upload your PDFs and ask questions about them using LLaMA 3 on Groq Cloud.")

# Initialize
pipeline = RAGPipeline()
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

uploaded_files = st.file_uploader("Upload PDF files", type="pdf", accept_multiple_files=True)

if uploaded_files:
    pdf_paths = []
    for uploaded_file in uploaded_files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded_file.read())
            pdf_paths.append(tmp.name)
            logging.info(f"Uploaded file saved to {tmp.name}")

    with st.spinner("Indexing PDFs..."):
        start = time.time()
        try:
            pipeline.load_and_index_pdfs(pdf_paths)
            indexing_time = time.time() - start
            st.metric("⏱️ Indexing Time", f"{indexing_time:.2f} sec")
            st.success(f"Indexing complete! Detected domain: **{pipeline.detected_domain}**")
            logging.info(f"Indexing complete. Domain: {pipeline.detected_domain}. Time: {indexing_time:.2f}s")
        except Exception as e:
            logging.error(f"Error during indexing: {e}")
            st.error("Error during indexing.")

    top_k = st.slider("How many chunks to retrieve?", 1, 5, 3)
    question = st.text_input("Ask a question about the documents:")

    if question:
        with st.spinner("Generating answer..."):
            start = time.time()
            try:
                top_chunks = pipeline.query(question, top_k=top_k)
                retrieval_time = time.time() - start
                st.metric("🔍 Retrieval Time", f"{retrieval_time:.2f} sec")
                logging.info(f"Question: {question} | Retrieved {len(top_chunks)} chunks in {retrieval_time:.2f}s")

                if top_chunks:
                    context = "\n\n".join([f"{c['text']}" for c in top_chunks])
                    prompt = f"""You are an expert in **{pipeline.detected_domain}** documents.

Use the following document snippets to answer the question **accurately and concisely**:

### Document Context:
{context}

### Question:
{question}

### Answer:"""

                    # Add to chat history
                    st.session_state.chat_history.append({"role": "user", "content": prompt})

                    response = client.chat.completions.create(
                        model="llama3-70b-8192",
                        messages=st.session_state.chat_history,
                        temperature=0.3
                    )

                    answer = response.choices[0].message.content
                    st.session_state.chat_history.append({"role": "assistant", "content": answer})

                    st.markdown("### Answer")
                    st.markdown(answer)
                    logging.info(f"Answer generated for question: {question}")
                else:
                    st.warning("No relevant chunks found.")
                    logging.warning(f"No relevant chunks found for question: {question}")
            except Exception as e:
                logging.error(f"Error during answer generation: {e}")
                st.error("Error during answer generation.")

    pipeline.cleanup()
    logging.info("Pipeline cleanup complete.")
