#%%
# -----------------------------------------------------------------------------
# Environment and core dependencies
# -----------------------------------------------------------------------------
import os
import sys

# -----------------------------------------------------------------------------
# Document ingestion
# -----------------------------------------------------------------------------
from pypdf import PdfReader  # Parsing PDF structure and extracting page-level text.

# -----------------------------------------------------------------------------
# Vector database (Chroma) and embedding protocol
# -----------------------------------------------------------------------------
import chromadb  # Client bindings for Chroma (local or cloud-backed vector stores).
from chromadb.api.types import EmbeddingFunction, Documents, Embeddings  # Protocol types enabling a pluggable embedding layer.

# -----------------------------------------------------------------------------
# OpenAI client (LLM + embeddings)
# -----------------------------------------------------------------------------
from openai import OpenAI  # Unified SDK for chat completions and embedding endpoints.

# -----------------------------------------------------------------------------
# Text splitting utilities
# -----------------------------------------------------------------------------
from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,          # Hierarchical splitting that respects coarse-to-fine separators.
    SentenceTransformersTokenTextSplitter    # Token-aware splitter suitable for embedding-friendly chunking.
)

# -----------------------------------------------------------------------------
# Numerical computing and tabular inspection
# -----------------------------------------------------------------------------
import numpy as np    # Dense numerical arrays for post-processing vectors and similarity calculations.
import pandas as pd   # Lightweight tabular views to audit metadata and chunk statistics during ingestion.

# -----------------------------------------------------------------------------
# Typing aids for clarity and static checks
# -----------------------------------------------------------------------------
from typing import Optional  # Explicit optionality annotations to document nullable values.

# -----------------------------------------------------------------------------
# Environment loading
# -----------------------------------------------------------------------------
from dotenv import load_dotenv, dotenv_values  # Utilities for .env configuration management.
_ = load_dotenv()  # Loading environment variables at import time to simplify downstream configuration access.

from tqdm import tqdm # To show the progress
import uuid # generate uniques ids

#%%
# -----------------------------------------------------------------------------
# Model configuration
# -----------------------------------------------------------------------------
# LLM_MODEL designates the conversational model used for text generation and reasoning.
# EMB_MODEL designates the embedding model used for transforming text into dense vector
# representations suitable for similarity search and retrieval.
LLM_MODEL = "gpt-4.1-nano"           
EMB_MODEL = "text-embedding-3-small"
#%%
# -----------------------------------------------------------------------------
# Environment variable resolution
# -----------------------------------------------------------------------------
# Reading sensitive credentials and configuration values from the process environment.
# This design avoids hardcoding secrets in source code and facilitates reproducibility
# across different deployment contexts (e.g., local development vs. cloud execution).
CHROMA_TENANT_API = os.getenv("CHROMA_TENANT_API")   
CHROMA_API        = os.getenv("CHROMA_API")         
OPENAI_API        = os.getenv("OPENAI_API")          
CHROMA_DATABASE   = os.getenv("CHROMA_DATABASE")    


#%%
# --- Generation parameters ----------------------------------------------------
# Using explicit decoding hyperparameters to ensure determinism across runs.
# TEMPERATURE controls stochasticity during decoding; MAX_TOKENS bounds generation length and cost.
TEMPERATUER = 0.7
MAX_TOKENS = 100
#%%
# -----------------------------------------------------------------------------
# Client initialization
# -----------------------------------------------------------------------------
# Instantiating the OpenAI client with the provided API key. This client unifies
# access to both chat-completion endpoints (for generative tasks) and embedding
# endpoints (for vectorization).
llm_client = OpenAI(api_key=OPENAI_API)

# Instantiating the Chroma Cloud client. The parameters explicitly bind the session
# to a tenant, a logical database within that tenant, and an API key authorizing access.
# This client exposes methods for collection management, vector insertion, and retrieval.
chroma = chromadb.CloudClient(
    tenant=CHROMA_TENANT_API,
    database=CHROMA_DATABASE,
    api_key=CHROMA_API
)
#%%
# -----------------------------------------------------------------------------
# Custom embedding adapter
# -----------------------------------------------------------------------------
# This class serves as a bridge between OpenAI’s embedding service and Chroma’s
# embedding interface. Chroma expects any embedding provider to implement the
# EmbeddingFunction protocol, which defines how raw text is converted into
# dense vector representations. By wrapping the OpenAI client in this adapter,
# the workflow can precompute vectors on demand while keeping the interface
# consistent and modular.
class OpenAIEmbeddingFunction(EmbeddingFunction):
    def __init__(self, client: OpenAI, model: str):
        # Persisting a reference to the OpenAI client. This object handles network
        # communication with the OpenAI API, including authentication and request routing.
        self.client = client

        # Recording the embedding model identifier. This ensures that every call
        # to the adapter uses a consistent embedding space, which is critical for
        # meaningful similarity comparisons in vector search.
        self.model = model

    def __call__(self, inputs: Documents) -> Embeddings:
        # Step 1: Normalize inputs.
        # Chroma may pass either a single string or a list of strings. To simplify
        # processing, all inputs are coerced into a list so that batch requests can
        # be made uniformly.
        if isinstance(inputs, str):
            inputs = [inputs]

        # Step 2: Request embeddings.
        # The normalized list of texts is sent to the OpenAI embeddings endpoint.
        # The chosen model dictates the dimensionality and semantic properties
        # of the returned vectors.
        resp = self.client.embeddings.create(model=self.model, input=inputs)

        # Step 3: Extract embeddings.
        # The response contains an embedding for each input text. These embeddings
        # are high-dimensional float vectors. They are returned as plain Python
        # lists, which are directly consumable by Chroma’s storage and retrieval
        # mechanisms without further conversion.
        return [item.embedding for item in resp.data]
#%%
# -----------------------------------------------------------------------------
# Embedding function instantiation
# -----------------------------------------------------------------------------
# Creating a concrete instance of the embedding adapter. The instance binds together:
# - The OpenAI client (responsible for network communication and authentication).
# - The embedding model identifier (ensuring all texts are projected into the same
#   semantic vector space).
#
# This object can now be invoked as a callable to transform arbitrary text into
# embeddings, enabling explicit precomputation of vectors before insertion into
# the Chroma collection.
emb_fn = OpenAIEmbeddingFunction(llm_client, EMB_MODEL)
#%%
# -----------------------------------------------------------------------------
# LLM connectivity test
# -----------------------------------------------------------------------------
# Performing a minimal chat completion request to confirm that the OpenAI client
# is functional and correctly authenticated. This interaction serves as a live
# connectivity probe before building more complex pipelines.
#
# Key parameters:
# - model: specifies the conversational engine used to generate responses.
# - temperature: controls randomness in decoding; lower values yield more deterministic answers.
# - max_tokens: sets a hard cap on the number of tokens produced, limiting both
#   verbosity and API cost.
# - messages: structures the conversational context. A "system" role defines the
#   assistant’s behavior, while a "user" role introduces the actual query.
chat = llm_client.chat.completions.create(
    model=LLM_MODEL,
    temperature=TEMPERATUER,
    max_tokens=MAX_TOKENS,
    messages=[
        {"role": "system", "content": "You are an expert in geography."},
        {"role": "user", "content": "What is the capital of Spain?"}
    ]
)
#%%
# --- Validation cell: LLM connectivity test -----------------------------------
# Using a direct print of the assistant's reply to confirm that the model responds as expected.
# print(chat.choices[0].message.content)
#%%
# --- Creating or retrieving a Chroma collection -------------------------------
# Defining a symbolic name for the collection, binding stored documents and their embeddings
# under a reproducible identifier that can be referenced across experiments.
collection_name = "document_qa_collection"

# Creating (or retrieving) the collection from Chroma. Since embeddings are precomputed
# explicitly, no embedding_function is attached at collection instantiation.
collection = chroma.get_or_create_collection(
    name=collection_name,
)
# Printing the collection name to confirm that the resource is available and accessible.
# print("Colección lista:", collection.name)
# %%
# -----------------------------------------------------------------------------
# PDF ingestion
# -----------------------------------------------------------------------------
# Reading the Spanish Constitution PDF page by page and concatenating all text into
# a single string. This unified corpus will later be segmented into smaller chunks
# for embedding and retrieval purposes.
reader = PdfReader("data/constitución_española.pdf")

# Extract text from each page, stripping leading/trailing whitespace for cleanliness.
pdf_texts = [page.extract_text().strip() for page in reader.pages if page.extract_text()]

# Join all page-level text blocks into a single string separated by double newlines.
full_text = "\n\n".join(pdf_texts)

#%%
# Quick sanity check: print length in characters and number of pages processed.
# print(f"Extracted {len(pdf_texts)} pages, total characters: {len(full_text)}")


# %%
# -----------------------------------------------------------------------------
# First-pass segmentation: character-level chunking (no overlap)
# -----------------------------------------------------------------------------
# Using a coarse-to-fine, separator-aware strategy to split the unified corpus
# into ~1000-character segments. This stage introduces *no* overlap because the
# goal is to produce stable, deterministic blocks that will later be re-chunked
# at the token level (where overlap will be introduced for retrieval robustness).
#
# Rationale:
# - Separators are ordered from strongest to weakest to preserve semantic
#   boundaries when possible (paragraphs, lines, sentences, whitespace, fallback).
# - chunk_size=1000 provides a manageable payload for subsequent token-aware
#   chunking and embedding; actual sizes will vary slightly due to separator cuts.
# - chunk_overlap=0 keeps this pass purely partitioning; overlap will be handled
#   in the token-level splitter to optimize recall without inflating storage.
character_splitter = RecursiveCharacterTextSplitter(
    separators=["\n\n", "\n", ". ", " ", ""],
    chunk_size=1000,
    chunk_overlap=0,
)

# Producing a list of character-level segments. Each element is a contiguous
# span of the original text respecting the separator hierarchy above.
character_split_texts = character_splitter.split_text(full_text)

# Lightweight diagnostics to validate distribution and size characteristics.
num_chunks = len(character_split_texts)
lengths = [len(t) for t in character_split_texts]
avg_len = sum(lengths) / num_chunks if num_chunks else 0

print(f"[character-chunks] count={num_chunks}, avg_len={avg_len:.1f}, "
      f"min_len={min(lengths) if lengths else 0}, max_len={max(lengths) if lengths else 0}")


print("------------------------------------------------------")
preview_df = pd.DataFrame({
    "chunk_id": list(range(min(5, num_chunks))),
    "length": lengths[:5],
    "text_preview": [t[:120].replace("\n", " ") + ("..." if len(t) > 120 else "") for t in character_split_texts[:5]],
})
print(preview_df.to_string(index=False))


# %%
# -----------------------------------------------------------------------------
# Second-pass segmentation: token-level chunking (with overlap)
# -----------------------------------------------------------------------------
# Taking the output from the character-level split and re-segmenting it into
# token-sized windows suitable for embedding. This ensures that:
# - Each chunk respects the embedding model’s maximum context (smaller than LLM context).
# - A small overlap between consecutive chunks preserves continuity of meaning,
#   which improves recall when retrieving similar passages later.
#
# Configuration:
# - tokens_per_chunk=256 ensures each chunk is concise and efficient for the
#   embedding model, balancing semantic coverage with storage.
# - chunk_overlap=35 provides shared context across neighbors, mitigating the
#   risk of losing key information at boundaries.
token_splitter = SentenceTransformersTokenTextSplitter(
    chunk_overlap=35,
    tokens_per_chunk=256,
)

# Apply token-level splitting to each character-level segment.
token_split_texts = []
for text in character_split_texts:
    token_split_texts.extend(token_splitter.split_text(text))

# Diagnostics: count chunks and show average token-level length (approximate).
num_token_chunks = len(token_split_texts)
lengths = [len(t.split()) for t in token_split_texts]  # crude word-based proxy for inspection
avg_len = sum(lengths) / num_token_chunks if num_token_chunks else 0

print(f"[token-chunks] count={num_token_chunks}, avg_wordlen≈{avg_len:.1f}, "
      f"min_words={min(lengths) if lengths else 0}, max_words={max(lengths) if lengths else 0}")

# preview: first few chunks to verify integrity.
for i, chunk in enumerate(token_split_texts[:3]):
    print(f"\n--- chunk {i} ---\n{chunk[:300]}{'...' if len(chunk) > 300 else ''}")



# %%
# -----------------------------------------------------------------------------
# Embedding precomputation and ingestion into Chroma Cloud
# -----------------------------------------------------------------------------
# The objective is to transform the finalized token-level chunks into dense vectors
# and persist them in the target Chroma collection. Each record is written with:
# - ids: globally unique identifiers for deduplication and deterministic retrieval.
# - documents: the raw text chunk (as stored content).
# - metadatas: lightweight descriptors to facilitate filtering and audit.
# - embeddings: precomputed vectors aligned with the chosen embedding model.
#
# Batching is used to respect API rate limits and maintain memory efficiency.
BATCH_SIZE = 128  # Balance between throughput and memory usage.

def batched(iterable, n):
    """Yield successive n-sized slices from an iterable without copying the full list unnecessarily."""
    for i in range(0, len(iterable), n):
        yield iterable[i:i + n]

# Build metadata and identifiers for every chunk.
metadatas_all = [
    {
        "source": "constitución_española.pdf",
        "chunk_index": i,
        "char_len": len(text),
        "splitter": "SentenceTransformersTokenTextSplitter:256/overlap35"
    }
    for i, text in enumerate(token_split_texts)
]
ids_all = [f"es-const-{uuid.uuid4()}" for _ in token_split_texts]

# Validation: ensure aligned arrays.
assert len(ids_all) == len(token_split_texts) == len(metadatas_all), "Parallel arrays must be same length."

# Ingest in batches with progress bar.
total_chunks = len(ids_all)
with tqdm(total=total_chunks, desc="Ingesting chunks into Chroma", unit="chunk") as pbar:
    for ids_batch, docs_batch, metas_batch in zip(
        batched(ids_all, BATCH_SIZE),
        batched(token_split_texts, BATCH_SIZE),
        batched(metadatas_all, BATCH_SIZE),
    ):
        # Precompute embeddings for the current batch.
        emb_batch = emb_fn(docs_batch)

        # Add batch into Chroma collection.
        collection.add(
            ids=ids_batch,
            documents=docs_batch,
            metadatas=metas_batch,
            embeddings=emb_batch,
        )

        # Update progress bar with number of chunks just processed.
        pbar.update(len(ids_batch))

print(f"[done] collection count -> {collection.count()}")


#%%
# -----------------------------------------------------------------------------
# Query expansion via LLM (Multi-Query)
# -----------------------------------------------------------------------------
# Given a user question, generate three semantically diverse paraphrases that
# preserve the original intent. The function returns the original question plus
# three new variants, all phrased differently but targeting the same meaning.
#
# This diversification increases recall during retrieval by probing the vector
# database with multiple semantic formulations of the same question.
def expand_queries(user_question: str) -> list[str]:
    prompt = (
        "You will receive a user question. Produce exactly three alternative phrasings that preserve the same intent, "
        "diversify vocabulary, and avoid adding or removing constraints.\n\n"
        "Return the three alternatives as plain text, one per line, with no numbering and no extra commentary.\n\n"
        f"User question:\n{user_question}"
    )

    resp = llm_client.chat.completions.create(
        model=LLM_MODEL,
        temperature=0.6,   
        max_tokens=200,
        messages=[
            {"role": "system", "content": "You generate concise, faithful paraphrases for retrieval."},
            {"role": "user", "content": prompt},
        ],
    )

    # Parse the output into lines, ignoring empty strings.
    lines = [ln.strip() for ln in resp.choices[0].message.content.split("\n") if ln.strip()]

    # Limit to three alternatives (truncate if more are returned).
    alts = lines[:3]

    # Return the original user question followed by the generated alternatives.
    return [user_question] + alts

#%%
user_question = "¿Cuál es la lengua oficial del Estado y qué establece la Constitución sobre ella?"
queries = expand_queries(user_question)

# %%
print(queries)
print(len(queries))

# %%
# -----------------------------------------------------------------------------
# Precompute embeddings for the expanded queries (no retrieval yet)
# -----------------------------------------------------------------------------
# Given the list of queries produced by `expand_queries`, compute one embedding
# per query using the same embedding model that was used for documents. Using a
# shared embedding space is essential for meaningful nearest-neighbor search.
# Sanity check: ensure there is at least one query to embed.
assert len(queries) > 0, "No queries to embed. Did expand_queries() return an empty list?"
# Compute embeddings in a single batch; the adapter returns a list of float lists.
query_embeddings = emb_fn(queries)
# Lightweight diagnostics: confirm 1:1 alignment and report vector dimensionality.
assert len(query_embeddings) == len(queries), "Mismatch between queries and embeddings."
emb_dim = len(query_embeddings[0]) if query_embeddings else 0
print(f"[queries] count={len(queries)}, embedding_dim={emb_dim}")
# structuring results 
query_vectors = [
    {"query": q, "embedding": vec}
    for q, vec in zip(queries, query_embeddings)
]
# previewing and verifying content without flooding the console.
for i, item in enumerate(query_vectors):
    snippet = item["query"][:120].replace("\n", " ")
    print(f"- q{i}: {snippet}{'...' if len(item['query']) > 120 else ''}")

# %%
# -----------------------------------------------------------------------------
# Multi-query retrieval using precomputed embeddings (single batched call)
# -----------------------------------------------------------------------------
# Using the four query embeddings at once to retrieve the top-2 passages per query.
# Chroma returns parallel lists per issued query; results are normalized into a flat list.

# Defensive checks: embeddings and queries must align 1:1.
assert len(query_embeddings) == len(queries), "Queries and embeddings must align."
# Number of retrivals we want
N_RESULTS = 5

# Issue a single batched similarity search: one embedding per expanded query.
retrieval = collection.query(
    query_embeddings=query_embeddings,           # List[List[float]] (e.g., 4 vectors: original + 3 variants)
    n_results=N_RESULTS,                                 # Two hits per query → ~8 raw hits total
    include=["documents", "metadatas", "distances"],
)

# Normalize results into a flat structure for downstream processing.
raw_hits = []
for qi, q_text in enumerate(queries):
    ids   = retrieval.get("ids", [[]])[qi]
    docs  = retrieval.get("documents", [[]])[qi]
    metas = retrieval.get("metadatas", [[]])[qi]
    dists = retrieval.get("distances", [[]])[qi]

    for i, doc_id in enumerate(ids):
        raw_hits.append({
            "id": doc_id,           # unique identifier of the retrieved chunk
            "text": docs[i],        # retrieved passage text
            "metadata": metas[i],   # provenance (e.g., chunk_index, source)
            "distance": dists[i],   # similarity distance (lower is closer)
            "from_query": q_text,   # which expanded query produced this hit
        })

raw_hits.sort(key=lambda h: h["distance"]) # sort by distance 
print(f"[retrieval] queries={len(queries)}, per_query_k={N_RESULTS}, total_hits={len(raw_hits)}")

# %%
# -----------------------------------------------------------------------------
# Deduplicate by id and keep top-N closest hits
# -----------------------------------------------------------------------------
TOP_N = 8
seen = set()
unique_hits = []
for h in raw_hits:
    if h["id"] in seen:
        continue
    seen.add(h["id"])
    unique_hits.append(h)

# Keep only the TOP_N after deduplication
top_hits = unique_hits[:TOP_N]
print(f"total_unique={len(unique_hits)}, top_used={len(top_hits)}")

# %%
# -----------------------------------------------------------------------------
# Pretty-print top_hits in full detail
# -----------------------------------------------------------------------------
for i, h in enumerate(top_hits, 1):
    print(f"\n[Hit {i}]")
    print(f"  id        : {h['id']}")
    print(f"  distance  : {h['distance']:.4f}")
    print(f"  from_query: {h['from_query']}")
    print(f"  metadata  : {h['metadata']}")
    print("  text      :")
    print(h["text"])
    print("-" * 80)

# %%
# -----------------------------------------------------------------------------
# Build grounded context from top hits
# -----------------------------------------------------------------------------
# Constructing a compact, traceable context block from the selected passages.
# Each passage is preceded by a header carrying provenance (id and chunk_index),
# enabling later citation and auditability in the final answer.
def build_context_block(hits: list[dict], max_chars: int = 12000) -> str:
    parts = []
    total = 0
    for i, h in enumerate(hits, 1):
        header = f"[Passage {i} | id={h['id']} | chunk_index={h['metadata'].get('chunk_index', 'NA')}]"
        body = (h["text"] or "").strip()
        segment = f"{header}\n{body}\n"
        if total + len(segment) > max_chars:
            break
        parts.append(segment)
        total += len(segment)
    return "\n---\n".join(parts)

# Materialize the context from the already prepared `top_hits`.
context_block = build_context_block(top_hits)
print("\n[context preview]\n")
print(context_block[:1200] + ("\n...\n" if len(context_block) > 1200 else ""))

# %%
# -----------------------------------------------------------------------------
# Answer strictly using provided context
# -----------------------------------------------------------------------------
# The answering prompt enforces grounding: the model must rely only on the
# supplied passages. If the context is insufficient, the answer should explicitly
# acknowledge that limitation. Output is requested in Spanish to match the use case.
def answer_with_context(user_question: str, context_block: str) -> str:
    system_msg = (
        "Responde estrictamente usando ÚNICAMENTE los pasajes de contexto proporcionados. "
        "Si la información necesaria no está en el contexto, indica claramente que no se puede "
        "determinar la respuesta a partir del contexto. Cita entre corchetes el índice del pasaje "
        "relevante cuando corresponda (por ejemplo, [Passage 2]). Responde en español, de forma concisa y precisa."
    )
    user_msg = (
        f"Pregunta:\n{user_question}\n\n"
        f"Pasajes de contexto:\n{context_block}\n\n"
        "Instrucciones: Usa solo el contexto anterior. Si no alcanza para responder con seguridad, dilo explícitamente."
    )
    resp = llm_client.chat.completions.create(
        model=LLM_MODEL,
        temperature=0.1,
        max_tokens=400,
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ],
    )
    return resp.choices[0].message.content.strip()

# Example: synthesize the final answer grounded in `top_hits`.
final_answer = answer_with_context(
    user_question=user_question,   # reuse the question you set earlier
    context_block=context_block
)

print("\n[final answer]\n")
print(final_answer)

# %%
