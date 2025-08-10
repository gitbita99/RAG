
---

## **README.md**


# Modular RAG Pipeline with Persistent ChromaDB

This project implements a **Retrieval-Augmented Generation (RAG)** pipeline in a modular structure using **LangChain**, **HuggingFace embeddings**, and **ChromaDB** with persistence.  
The goal: Efficiently query large PDFs (e.g., 300+ pages) without reprocessing them every run.

---

## **📂 Project Structure**


project/
│── chunking.py        # Loads and splits PDF into text chunks
│── embedding.py       # Creates / loads persistent Chroma vectorstore
│── rag\_pipeline.py    # Main pipeline for querying
│── requirements.txt   # Python dependencies
│── data/              # Store your PDF files here
│── vectorstore/       # Auto-generated persistent ChromaDB folder
│── README.md          # This file


---

## **⚡ Features**
- **Modular code** for better maintainability.
- **Persistent vectorstore** (ChromaDB) for instant reloads after the first run.
- **LangChain + HuggingFace** embeddings.
- Optimized for **large documents**.

---

## **📦 Installation**
1. Clone the repository:
   ```bash
   git clone https://github.com/<your-username>/<your-repo>.git
   cd <your-repo>
   ````

2. Create a virtual environment & install dependencies:

   ```bash
   python -m venv venv
   source venv/bin/activate  # macOS/Linux
   venv\Scripts\activate     # Windows

   pip install -r requirements.txt
   ```

---

## **🛠 Usage**

1. Place your PDF in the `data/` folder (e.g., `data/document.pdf`).

2. Run the pipeline:

   ```bash
   python rag_pipeline.py
   ```

3. First run:

   * PDF will be split into chunks.
   * Embeddings will be created & stored in `vectorstore/`.

4. Subsequent runs:

   * Loads from `vectorstore/` instantly without re-embedding.

---

## **🗂 Requirements**

```
langchain
langchain-community
langchain-huggingface
chromadb
pypdf
```

---

## **🖥 Example Output**

```
✅ Loading existing vectorstore...
Q: What is the main topic of the document?
A: The PDF discusses...
```

---

## **📜 License**

MIT License – free to use and modify.

---
