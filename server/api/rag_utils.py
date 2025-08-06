import os
import logging
import json
import time
from typing import List, Dict, Optional, Tuple, Union
from pathlib import Path
import numpy as np
import pandas as pd
from datetime import datetime
# Core packages
from sentence_transformers import SentenceTransformer

# Hugging Face for text generation
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import torch

# NLTK for simple tokenization
import nltk
try:
    nltk.download('punkt', quiet=True)
except:
    pass

# LangChain for text splitting
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

# BM25 for keyword search
from rank_bm25 import BM25Okapi

# Pinecone (you can also replace this with FAISS for fully local)
from pinecone import Pinecone, ServerlessSpec

# Evaluation
from sklearn.metrics.pairwise import cosine_similarity

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

print("✅ All libraries imported (Hugging Face only)!")


# Configuration
CONFIG = {
    "embedding_model": "BAAI/bge-small-en",
    "generation_model": "Qwen/Qwen2.5-3B-Instruct",  
    
    "index_name": "hybrid-rag-langchain",
    
    # LangChain RecursiveCharacterTextSplitter parameters
    "chunk_size": 300,       # characters per chunk
    "chunk_overlap": 50,     # overlapping characters (20% overlap)
    "separators": ["\n\n", "\n", " ", ""],  # Hierarchy of separators
    "retrieval_k": 10
}

# Set your Pinecone API key (only one API key needed now!)
PINECONE_API_KEY = "pcsk_6m2PRg_1qqfqLoS7ZEfXyacwJzrjwkKaQUA5aW3VQjV7wVoMfLH7S8MYZPG2sD5QaVeSE"

# Check if GPU is available
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🔧 Using device: {device}")

# Verify API key
if not PINECONE_API_KEY or PINECONE_API_KEY == "your_pinecone_api_key_here":
    print("⚠️ Please set your PINECONE_API_KEY")
else:
    print("✅ Pinecone API key configured")

print(f"🎯 Models selected:")
print(f"  Embedding: {CONFIG['embedding_model']}")
print(f"  Generation: {CONFIG['generation_model']}")



# Global variables for lazy loading
embedding_model = None
embedding_dimension = None
text_splitter = None
generator = None
pc = None
index = None
documents_corpus = []
document_metadata = []
bm25_index = None

def initialize_models():
    """Lazy load models only when needed"""
    global embedding_model, embedding_dimension, text_splitter, generator, pc, index
    
    if embedding_model is None:
        print(f"🔄 Loading embedding model: {CONFIG['embedding_model']}")
        embedding_model = SentenceTransformer(CONFIG['embedding_model'])
        embedding_dimension = embedding_model.get_sentence_embedding_dimension()
        print(f"✅ Embedding model loaded! Dimension: {embedding_dimension}")
    
    if text_splitter is None:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CONFIG['chunk_size'],
            chunk_overlap=CONFIG['chunk_overlap'],
            separators=CONFIG['separators'],
            length_function=len,
            is_separator_regex=False
        )
        print(f"✅ LangChain text splitter initialized")
    
    if generator is None:
        print(f"🔄 Loading generation model: {CONFIG['generation_model']}")
        try:
            generator = pipeline(
                "text-generation",
                model=CONFIG['generation_model'],
                device=0 if device == "cuda" else -1,
                torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                max_length=1024,
                do_sample=True,
                temperature=0.8,
                top_p=0.9,
                pad_token_id=50256,
                trust_remote_code=True
            )
            print("✅ Text generation model loaded!")
        except Exception as e:
            print(f"⚠️ Trying CPU configuration: {e}")
            try:
                generator = pipeline(
                    "text-generation",
                    model=CONFIG['generation_model'],
                    device=-1,
                    max_length=512,
                    trust_remote_code=True
                )
                print("✅ Text generation model loaded on CPU!")
            except Exception as e2:
                print(f"❌ Model loading failed: {e2}")
                generator = None
    
    if pc is None:
        print("🔄 Initializing Pinecone...")
        pc = Pinecone(api_key=PINECONE_API_KEY)
        
        # Check if index exists
        existing_indexes = [idx.name for idx in pc.list_indexes()]
        
        if CONFIG['index_name'] not in existing_indexes:
            print(f"🔄 Creating Pinecone index: {CONFIG['index_name']}")
            pc.create_index(
                name=CONFIG['index_name'],
                dimension=embedding_dimension,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1")
            )
            print("⏳ Waiting for index to be ready...")
            time.sleep(10)
        
        index = pc.Index(CONFIG['index_name'])
        print(f"✅ Connected to Pinecone index: {CONFIG['index_name']}")

print("✅ RAG utilities module loaded (models will be initialized on first use)")


def chunk_documents(documents: List[str], doc_ids: Optional[List[str]] = None) -> List[Dict]:
    """
    Chunk documents using LangChain RecursiveCharacterTextSplitter
    """
    # Initialize models if needed
    initialize_models()
    
    if doc_ids and len(doc_ids) != len(documents):
        raise ValueError("doc_ids length must match documents length")
    
    all_chunks = []
    
    for i, doc_text in enumerate(documents):
        doc_id = doc_ids[i] if doc_ids else f"doc_{i}"
        
        # Create LangChain Document object
        doc = Document(
            page_content=doc_text,
            metadata={"doc_id": doc_id, "original_length": len(doc_text)}
        )
        
        # Split using LangChain text splitter
        split_docs = text_splitter.split_documents([doc])
        
        # Convert to our format
        for chunk_idx, split_doc in enumerate(split_docs):
            chunk_text = split_doc.page_content
            word_count = len(chunk_text.split())
            char_count = len(chunk_text)
            
            all_chunks.append({
                "id": f"{doc_id}_chunk_{chunk_idx}",
                "text": chunk_text,
                "metadata": {
                    "doc_id": doc_id,
                    "chunk_idx": chunk_idx,
                    "total_chunks": len(split_docs),
                    "word_count": word_count,
                    "char_count": char_count,
                    "original_length": len(doc_text)
                }
            })
    
    # Calculate statistics
    total_chunks = len(all_chunks)
    avg_words = sum(c['metadata']['word_count'] for c in all_chunks) / total_chunks if total_chunks else 0
    avg_chars = sum(c['metadata']['char_count'] for c in all_chunks) / total_chunks if total_chunks else 0
    
    print(f"✅ Created {total_chunks} LangChain chunks from {len(documents)} documents")
    print(f"📊 Average per chunk: {avg_words:.1f} words, {avg_chars:.0f} characters")
    
    return all_chunks

def vectorize_chunks(chunks: List[Dict]) -> List[Dict]:
    """Generate embeddings for chunks using BAAI/bge-small-en"""
    # Initialize models if needed
    initialize_models()
    
    if not chunks:
        return []
    
    print(f"🔄 Generating embeddings for {len(chunks)} chunks...")
    start_time = time.time()
    
    # Extract texts
    texts = [chunk["text"] for chunk in chunks]
    
    # Generate embeddings in batch
    embeddings = embedding_model.encode(
        texts,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True  # Important for cosine similarity
    )
    
    # Add embeddings to chunks
    for chunk, embedding in zip(chunks, embeddings):
        chunk["embedding"] = embedding.tolist()
    
    total_time = time.time() - start_time
    print(f"✅ Embeddings generated successfully in {total_time:.2f} seconds")
    print(f"   Average time per chunk: {total_time/len(chunks):.3f}s")
    
    return chunks




def store_in_pinecone(chunks: List[Dict]) -> bool:
    """Store chunks in Pinecone vector database"""
    # Initialize models if needed
    initialize_models()
    
    if not chunks:
        return False
    
    try:
        print(f"🔄 Storing {len(chunks)} chunks in Pinecone...")
        start_time = time.time()
        
        vectors = []
        for chunk in chunks:
            vector = {
                "id": chunk["id"],
                "values": chunk["embedding"],
                "metadata": {
                    "text": chunk["text"][:1000],  # Pinecone metadata limit
                    "doc_id": chunk["metadata"]["doc_id"],
                    "chunk_idx": chunk["metadata"]["chunk_idx"],
                    "word_count": chunk["metadata"]["word_count"],
                    "char_count": chunk["metadata"]["char_count"]
                }
            }
            vectors.append(vector)
        
        # Batch upsert
        batch_size = 100
        for i in range(0, len(vectors), batch_size):
            batch = vectors[i:i + batch_size]
            index.upsert(vectors=batch)
        
        total_time = time.time() - start_time
        print(f"✅ Successfully stored in Pinecone in {total_time:.2f} seconds")
        return True
        
    except Exception as e:
        print(f"❌ Failed to store in Pinecone: {e}")
        return False




def build_bm25_index(chunks: List[Dict]) -> bool:
    """Build BM25 index for keyword-based retrieval"""
    global documents_corpus, document_metadata, bm25_index
    
    try:
        print("🔄 Building BM25 index...")
        start_time = time.time()
        
        # Store documents and metadata
        documents_corpus = [chunk["text"] for chunk in chunks]
        document_metadata = [
            {
                "id": chunk["id"],
                "doc_id": chunk["metadata"]["doc_id"],
                "chunk_idx": chunk["metadata"]["chunk_idx"]
            }
            for chunk in chunks
        ]
        
        # Tokenize documents for BM25
        tokenized_docs = [doc.lower().split() for doc in documents_corpus]
        
        # Build BM25 index
        bm25_index = BM25Okapi(tokenized_docs)
        
        total_time = time.time() - start_time
        print(f"✅ BM25 index built with {len(documents_corpus)} documents in {total_time:.2f} seconds")
        return True
        
    except Exception as e:
        print(f"❌ Failed to build BM25 index: {e}")
        return False


def semantic_search(query: str, top_k: int = 10) -> List[Dict]:
    """Semantic search using Pinecone"""
    # Initialize models if needed
    initialize_models()
    
    try:
        start_time = time.time()
        
        # Generate query embedding
        query_embedding = embedding_model.encode([query], normalize_embeddings=True)[0]
        
        # Search Pinecone
        results = index.query(
            vector=query_embedding.tolist(),
            top_k=top_k,
            include_metadata=True
        )
        
        # Format results
        semantic_results = []
        for match in results.matches:
            semantic_results.append({
                "id": match.id,
                "score": match.score,
                "text": match.metadata.get("text", ""),
                "doc_id": match.metadata.get("doc_id", ""),
                "source": "semantic"
            })
        
        search_time = time.time() - start_time
        print(f"🔍 Semantic search completed in {search_time:.3f}s")
        
        return semantic_results
        
    except Exception as e:
        print(f"❌ Semantic search failed: {e}")
        return []



def keyword_search(query: str, top_k: int = 10) -> List[Dict]:
    """Keyword-based search using BM25"""
    if bm25_index is None:
        print("⚠️ BM25 index not built")
        return []
    
    try:
        start_time = time.time()
        
        # Tokenize query
        query_tokens = query.lower().split()
        
        # Get BM25 scores
        scores = bm25_index.get_scores(query_tokens)
        
        # Get top-k results
        top_indices = np.argsort(scores)[::-1][:top_k]
        
        # Format results
        keyword_results = []
        for idx in top_indices:
            if scores[idx] > 0:  # Only include positive scores
                keyword_results.append({
                    "id": document_metadata[idx]["id"],
                    "score": float(scores[idx]),
                    "text": documents_corpus[idx],
                    "doc_id": document_metadata[idx]["doc_id"],
                    "source": "keyword"
                })
        
        search_time = time.time() - start_time
        print(f"🔍 Keyword search completed in {search_time:.3f}s")
        
        return keyword_results
        
    except Exception as e:
        print(f"❌ Keyword search failed: {e}")
        return []


def reciprocal_rank_fusion(semantic_results: List[Dict], keyword_results: List[Dict], k: int = 60) -> List[Dict]:
    """Combine semantic and keyword results using Reciprocal Rank Fusion"""
    
    # Create score dictionaries
    rrf_scores = {}
    
    # Add semantic search scores
    for rank, result in enumerate(semantic_results):
        doc_id = result["id"]
        rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + 1 / (k + rank + 1)
    
    # Add keyword search scores
    for rank, result in enumerate(keyword_results):
        doc_id = result["id"]
        rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + 1 / (k + rank + 1)
    
    # Create combined results
    all_results = {}
    
    # Add all results to dictionary
    for result in semantic_results + keyword_results:
        doc_id = result["id"]
        if doc_id not in all_results:
            all_results[doc_id] = result.copy()
            all_results[doc_id]["rrf_score"] = rrf_scores[doc_id]
            all_results[doc_id]["sources"] = [result["source"]]
        else:
            if result["source"] not in all_results[doc_id]["sources"]:
                all_results[doc_id]["sources"].append(result["source"])
    
    # Sort by RRF score
    fused_results = list(all_results.values())
    fused_results.sort(key=lambda x: x["rrf_score"], reverse=True)
    
    return fused_results

def generate_response_hf(query: str, context_docs: List[Dict], max_new_tokens: int = 800) -> str:
    """Generate response using Hugging Face model with retrieved context"""
    # Initialize models if needed
    initialize_models()
    
    if not context_docs:
        return "I couldn't find relevant information to answer your question."
    
    if generator is None:
        return "Text generation model not available. Please check model loading."
    
    start_time = time.time()
    
    # Prepare context (increased limits)
    context_parts = []
    for i, doc in enumerate(context_docs[:5]):  # Use top 5 docs
        context_parts.append(f"Context {i+1}: {doc['text'][:800]}")  # 800 chars per doc
    
    context = "\n".join(context_parts)
    
    # Clean, simple prompt that won't confuse the model
    prompt = f"""Context: {context}

Question: {query}

Answer:"""
    
    try:
        # Generate response
        response = generator(
            prompt,
            max_new_tokens=max_new_tokens,
            num_return_sequences=1,
            truncation=True,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=generator.tokenizer.eos_token_id if generator.tokenizer.eos_token_id else 50256
        )
        
        # Extract generated text
        generated_text = response[0]['generated_text']
        
        # Extract only the answer part (after "Answer:")
        if "Answer:" in generated_text:
            answer = generated_text.split("Answer:")[-1].strip()
        else:
            answer = generated_text[len(prompt):].strip()
        
        # Clean up the answer - remove any extra instructions
        if "You are an AI assistant" in answer:
            answer = answer.split("You are an AI assistant")[0].strip()
        if "Task:" in answer:
            answer = answer.split("Task:")[0].strip()
        
        # Allow longer responses (up to 10 lines)
        answer_lines = answer.split('\n')
        answer = '\n'.join(answer_lines[:10]) if len(answer_lines) > 1 else answer
        
        # Ensure minimum length
        if not answer or len(answer) < 20:
            return "I need more context to provide a comprehensive answer."
        
        generation_time = time.time() - start_time
        print(f"🤖 Response generation completed in {generation_time:.2f}s")
        
        return answer
        
    except Exception as e:
        print(f"❌ Generation failed: {e}")
        return "Sorry, I encountered an error while generating the response."


def evaluate_response_simple(query: str, response: str, context_docs: List[Dict]) -> Dict:
    """Simple evaluation without generating follow-up questions"""
    
    # Simple metrics
    response_length = len(response.split())
    context_used = len(context_docs)
    
    # Check if response contains key terms from query
    query_words = set(query.lower().split())
    response_words = set(response.lower().split())
    word_overlap = len(query_words.intersection(response_words)) / len(query_words) if query_words else 0
    
    # Semantic similarity between query and response
    try:
        query_embedding = embedding_model.encode([query])
        response_embedding = embedding_model.encode([response])
        semantic_similarity = cosine_similarity(query_embedding, response_embedding)[0][0]
    except:
        semantic_similarity = 0.0
    
    # Context relevance (average similarity between query and context docs)
    context_relevance = 0.0
    if context_docs:
        try:
            context_texts = [doc['text'] for doc in context_docs[:3]]
            context_embeddings = embedding_model.encode(context_texts)
            query_embedding = embedding_model.encode([query])
            relevance_scores = cosine_similarity(query_embedding, context_embeddings)[0]
            context_relevance = np.mean(relevance_scores)
        except:
            context_relevance = 0.0
    
    return {
        "query": query,
        "response": response,
        "response_length": response_length,
        "context_docs_used": context_used,
        "word_overlap_score": word_overlap,
        "semantic_similarity": float(semantic_similarity),
        "context_relevance": float(context_relevance),
        "overall_score": (word_overlap + semantic_similarity + context_relevance) / 3
    }



def rag_chat_flexible(query: str, method: str = "hybrid", top_k: int = 10) -> Dict:
    """
    Flexible RAG pipeline - supports any retrieval method
    
    Args:
        query: Question to ask
        method: "semantic", "keyword", "hybrid" (default: "hybrid")
        top_k: Number of chunks to retrieve
    """
    print(f"🔄 Processing query with {method.upper()} retrieval: {query}")
    overall_start = time.time()
    
    try:
        # Route to appropriate retrieval method
        retrieval_start = time.time()
        if method == "semantic":
            retrieved_docs = semantic_search(query, top_k)
        elif method == "keyword":
            retrieved_docs = keyword_search(query, top_k)
        elif method == "hybrid":
            sem_results = semantic_search(query, top_k)
            key_results = keyword_search(query, top_k)
            retrieved_docs = reciprocal_rank_fusion(sem_results, key_results)
        else:
            raise ValueError(f"Unknown method: {method}. Use 'semantic', 'keyword', or 'hybrid'")
        
        retrieval_time = time.time() - retrieval_start
        print(f"📚 Retrieval completed in {retrieval_time:.2f}s")
        
        # Generate response
        generation_start = time.time()
        response = generate_response_hf(query, retrieved_docs)
        generation_time = time.time() - generation_start
        
        evaluation = evaluate_response_simple(query, response, retrieved_docs)
        
        overall_time = time.time() - overall_start
        print(f"⏱️ Total RAG pipeline time: {overall_time:.2f}s")
        
        return {
            "query": query,
            "answer": response,
            "retrieval_method": method,
            "num_docs_retrieved": len(retrieved_docs),
            "retrieved_docs": retrieved_docs[:3],
            "doc_sources": list(set([doc["doc_id"] for doc in retrieved_docs])),
            "evaluation": evaluation,
            "timing": {
                "retrieval_time": retrieval_time,
                "generation_time": generation_time,
                "total_time": overall_time
            }
        }
        
    except Exception as e:
        print(f"❌ Chat pipeline failed: {e}")
        return {
            "query": query,
            "answer": f"Error: {e}",
            "retrieval_method": method,
            "num_docs_retrieved": 0,
            "retrieved_docs": [],
            "doc_sources": [],
            "evaluation": {},
            "timing": {}
        }


def add_documents_to_system(documents: List[str], doc_ids: Optional[List[str]] = None) -> bool:
    """
    Complete pipeline to add documents using LangChain text splitter
    """
    try:
        print(f"📄 Adding {len(documents)} documents with LangChain chunking...")
        ingestion_start = time.time()
        
        # Step 1: LangChain chunking
        chunk_start = time.time()
        chunks = chunk_documents(documents, doc_ids)
        chunk_time = time.time() - chunk_start
        print(f"✂️ Document chunking completed in {chunk_time:.2f}s")
        
        if not chunks:
            print("❌ No chunks created")
            return False
        
        # Step 2: Generate embeddings
        embedding_start = time.time()
        chunks_with_embeddings = vectorize_chunks(chunks)
        embedding_time = time.time() - embedding_start
        
        # Step 3: Store in Pinecone
        storage_start = time.time()
        pinecone_success = store_in_pinecone(chunks_with_embeddings)
        storage_time = time.time() - storage_start
        
        # Step 4: Build BM25 index for keyword search
        bm25_start = time.time()
        bm25_success = build_bm25_index(chunks_with_embeddings)
        bm25_time = time.time() - bm25_start
        
        success = pinecone_success and bm25_success
        
        if success:
            total_time = time.time() - ingestion_start
            print("✅ Successfully added all documents with LangChain chunking")
            print(f"📊 Total chunks in system: {len(chunks_with_embeddings)}")
            
            # Show chunking statistics
            total_words = sum(c['metadata']['word_count'] for c in chunks_with_embeddings)
            total_chars = sum(c['metadata']['char_count'] for c in chunks_with_embeddings)
            avg_words = total_words / len(chunks_with_embeddings)
            avg_chars = total_chars / len(chunks_with_embeddings)
            
            print(f"📊 Chunking stats: {avg_words:.1f} avg words, {avg_chars:.0f} avg chars per chunk")
            
            # Print timing summary
            print(f"\n⏱️ TIMING SUMMARY:")
            print(f"   Chunking: {chunk_time:.2f}s")
            print(f"   Embedding: {embedding_time:.2f}s")
            print(f"   Storage: {storage_time:.2f}s")
            print(f"   BM25 Indexing: {bm25_time:.2f}s")
            print(f"   Total Ingestion: {total_time:.2f}s")
            print(f"   Average embedding time per chunk: {embedding_time/len(chunks):.3f}s")
        else:
            print("❌ Failed to add some documents")
        
        return success
        
    except Exception as e:
        print(f"❌ Error adding documents: {e}")
        return False

# Load your parsed text files
def load_extracted_text_files(directory_path: str = "extracted_texts") -> List[str]:
    """Load text files from your document processing pipeline"""
    from pathlib import Path
    
    texts_dir = Path(directory_path)
    if not texts_dir.exists():
        print(f"❌ Directory not found: {directory_path}")
        return []
    
    # Find all extracted text files
    text_files = list(texts_dir.glob("*_extracted_text.txt"))
    
    documents = []
    doc_ids = []
    
    for file_path in text_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read().strip()
            
            if content:  # Only add non-empty files
                # Extract document name from filename
                doc_name = file_path.stem.replace("_extracted_text", "")
                
                documents.append(content)
                doc_ids.append(doc_name)
                
        except Exception as e:
            print(f"⚠️ Error reading {file_path}: {e}")
    
    print(f"📄 Loaded {len(documents)} text files from {directory_path}")
    if documents:
        total_words = sum(len(doc.split()) for doc in documents)
        avg_words = total_words / len(documents)
        print(f"📊 Average document length: {avg_words:.1f} words")
    
    return documents, doc_ids

# Example usage - load and add your documents
# Uncomment the lines below to test with your documents
"""
try:
    # Load your extracted text files
    my_documents, my_doc_ids = load_extracted_text_files("extracted_texts")
    
    if my_documents:
        # Add to RAG system with word-based chunking
        success = add_documents_to_system(my_documents, my_doc_ids)
        
        if success:
            print("🎉 Ready to chat with your documents!")
        else:
            print("❌ Setup failed")
    else:
        print("⚠️ No documents found. Please check the 'extracted_texts' directory.")
        
except Exception as e:
    print(f"❌ Error: {e}")
"""


def rag_chat(query: str) -> Dict:
    """Main RAG chat function for Django integration"""
    try:
        # Use hybrid retrieval (best performing method)
        result = rag_chat_flexible(query)
        return result
    except Exception as e:
        print(f"❌ Error in rag_chat: {e}")
        return {
            "answer": f"Sorry, I encountered an error: {str(e)}",
            "doc_sources": [],
            "evaluation": {"overall_score": 0},
            "num_docs_retrieved": 0
        }

def ingest_new_text_file(text_file_path: str, doc_id: str = None) -> str:
    """Ingest a single text file into the RAG system"""
    try:
        # Read the text file
        with open(text_file_path, 'r', encoding='utf-8') as f:
            text_content = f.read()
        
        # Generate doc_id if not provided
        if doc_id is None:
            doc_id = Path(text_file_path).stem
        
        # Add to RAG system
        success = add_documents_to_system([text_content], [doc_id])
        
        if success:
            print(f"✅ Successfully ingested: {doc_id}")
            return doc_id
        else:
            print(f"❌ Failed to ingest: {doc_id}")
            return None
            
    except Exception as e:
        print(f"❌ Error ingesting file {text_file_path}: {e}")
        return None

def ask_question(question: str):
    """Simple interface to ask questions about your aviation documents"""
    print(f"\n🔍 Question: {question}")
    print("="*60)
    
    # Use hybrid retrieval (best performing method)
    result = rag_chat_flexible(question)
    
    # Display results
    print(f"📋 Answer:")
    print(f"   {result['answer']}")
    print(f"\n📄 Source Documents: {', '.join(result['doc_sources'])}")
    print(f"📊 Confidence Score: {result['evaluation'].get('overall_score', 0):.3f}")
    print(f"📈 Retrieved {result['num_docs_retrieved']} relevant chunks")
    
    return None

# Test functions (uncomment to test)
# ask_question("What were the delays taken over two days coded?")
# ask_question("What GPS issues occurred?")
# ask_question("What maintenance was required?")