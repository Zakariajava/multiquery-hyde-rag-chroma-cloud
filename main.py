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
        {"role": "system", "content": "Eres un asistente útil y conciso."},
        {"role": "user", "content": "¿Cuál es la capital de españa?"}
    ]
)
#%%
# --- Validation cell: LLM connectivity test -----------------------------------
# Using a direct print of the assistant's reply to confirm that the model responds as expected.
print(chat.choices[0].message.content)
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
print("Colección lista:", collection.name)
# %%
