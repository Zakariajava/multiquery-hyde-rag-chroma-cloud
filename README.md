# Multi-Query + HyDE RAG with Chroma Cloud

This project implements a Retrieval-Augmented Generation (RAG) pipeline using **OpenAI models** for language generation and embeddings, combined with **Chroma Cloud** as the vector database. The workflow is designed to improve retrieval quality through **multi-query expansion** and optional.

---

## 🚀 Overview

Traditional RAG often struggles when a single user query does not align semantically with the way knowledge chunks are embedded. To address this, this project integrates:

1. **PDF ingestion and preprocessing**  
   - The Spanish Constitution (`constitución_española.pdf`) is loaded and converted into plain text.
   - Text is first split at character-level (approx. 1000 chars, no overlap) to ensure clean segmentation.
   - Then, a second token-level split (384 tokens with overlap) produces semantic chunks suitable for embeddings.

2. **Embedding with OpenAI**  
   - Embeddings are precomputed using OpenAI's `text-embedding-3-large`.
   - Batching is applied for efficient ingestion into Chroma Cloud.

3. **Chroma Cloud storage**  
   - Each chunk is stored with a unique ID, metadata (source, chunk index, splitter type), and its embedding vector.
   - This enables scalable similarity search.

4. **Multi-query expansion**  
   - Given a user question, the LLM generates three alternative paraphrases.   
   - This increases recall, since different formulations probe the database from multiple semantic angles.

5. **Multi-query retrieval**  
   - Each query is embedded and used to search Chroma (top-k per query).  
   - Results are flattened, sorted by **distance** (lower = closer), and filtered by a maximum distance threshold.  
   - Duplicates are removed by `id`.  
   - The top-N unique passages (e.g., 8) are selected.

6. **Context assembly**  
   - Selected passages are concatenated into a structured context block.  
   - Each passage is tagged with `[Passage n | id=... | chunk_index=...]` for traceability.  
   - This ensures the LLM answer can cite exactly which chunk supported its reasoning.

7. **LLM grounded answering**  
   - The user's original question plus the context block are passed to the LLM (`gpt-4.1`).  
   - The system prompt enforces grounding: *only use provided context, cite passages, and if context is insufficient, state it clearly*.  
   - Answers are returned concisely, in Spanish.

---

## 🔑 Key Techniques

- **Two-stage chunking**  
  Balances coverage (character-level split) and semantic integrity (token-level split with overlap).

- **Multi-query expansion**  
  Reduces dependency on the wording of the user's question by generating paraphrases.

- **Filtering + deduplication**  
  - `distance` threshold ensures noisy, low-similarity hits are discarded.  
  - Deduplication prevents the same chunk from dominating the context.  
  - Ranking guarantees that the most relevant passages are prioritized.

- **Grounded answering**  
  By citing passages explicitly, the LLM's response is verifiable and auditable.

---

## 📂 Project Structure

```
multiquery-hyde-rag-chroma-cloud/
│
├── data/
│   └── constitución_española.pdf   # Source document
│
├── main.py                          # Main RAG pipeline
├── README.md                        # Project documentation
└── .env                             # API keys (ignored in Git)
```

---

## ⚙️ Configuration

Environment variables (stored in `.env`):

- `OPENAI_API` → OpenAI API key  
- `CHROMA_API` → Chroma Cloud API key  
- `CHROMA_TENANT_API` → Chroma Cloud tenant  
- `CHROMA_DATABASE` → Chroma Cloud database name  

Model choices:

- `LLM_MODEL = "gpt-4.1"`  
- `EMB_MODEL = "text-embedding-3-large"` 

---

## 📊 Example

**User Question:**  
> *¿Cuál es la lengua oficial del Estado y qué establece la Constitución sobre ella?*

**Expanded Queries:**  
- ¿Qué reconoce la Constitución como lengua oficial del Estado y qué determina sobre ello?  
- ¿Qué idioma oficial reconoce el Estado y qué señala la Constitución acerca de su uso?  
- ¿Cuál es el idioma oficial del país y qué determina la Constitución respecto a este?  

**Retrieved Passages (top 2 shown):**  
```
[Passage 1 | chunk_index=8]
El castellano es la lengua española oficial del Estado. Todos los españoles tienen el deber de conocerla y el derecho a usarla...
```

**Final Answer:**  
> *La lengua oficial del Estado es el castellano. La Constitución establece que todos los españoles tienen el deber de conocerla y el derecho a usarla. Además, las demás lenguas españolas serán también oficiales en las respectivas comunidades autónomas [Passage 1].*

---