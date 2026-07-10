from collections import defaultdict

from langchain_huggingface import  HuggingFaceEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from pydantic import BaseModel
from typing import List

load_dotenv()

# This defines the expected shape of the LLM's output: a JSON object with one key, queries,
# containing a list of strings. Example valid instance:
# {"queries": ["What is RAG?", "Explain retrieval augmented generation", "How does RAG work in LLMs?"]}
class QueryVariations(BaseModel):
    queries: List[str]


def multi_query_generation(query):
    embedding_model = HuggingFaceEmbeddings(model="all-MiniLM-L6-v2")

    persistent_directory = "db/chroma_db"

    db = Chroma(
        persist_directory=persistent_directory,
        embedding_function=embedding_model,
        collection_metadata={"hnsw:space": "cosine"}
    )

    llm = ChatGroq(model="openai/gpt-oss-20b")

    # FIX: Forced json_mode to bypass Groq API 400 validation failures.
    # TODO: If migrating to ChatOpenAI (e.g., gpt-4o-mini), 'method="json_mode"'
    # can be safely removed as OpenAI handles structured outputs natively.
    llm_with_tools = llm.with_structured_output(QueryVariations,method="json_mode")

    prompt = f"""You are a helpful assistant that generates alternative search queries.
    Generate exactly 3 different variations of the original query that would help retrieve relevant documents from a vector database.
    
    Original query: {query}
    
    You must format your output strictly as a valid JSON object matching the requested schema. Do not add any plain conversational text outside the JSON structure.
    {{"queries": ["variation 1", "variation 2", "variation 3"]}}
"""
    #Must mention that follow the JSON structure

    response = llm_with_tools.invoke(prompt)
    query_variations = response.queries

    retriever = db.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={"k": 3,
                       "score_threshold":0.3})

    all_retrieval_results = []  # Store all results for RRF

    for i, query in enumerate(query_variations, 1):

        docs = retriever.invoke(query)
        # When you call retriever.invoke(query) for a single query, Chroma doesn't return documents in random or arbitrary order.
        # It returns them ordered by similarity score, from most similar to least similar.
        # That ordering is what becomes "position" in the RRF code.

        all_retrieval_results.append(docs)  # Store for RRF calculation


    def reciprocal_rank_fusion(chunk_lists, k=60):


        # Data structures for RRF calculation
        rrf_scores = defaultdict(float)  # Will store: {chunk_content: rrf_score}
        all_unique_chunks = {}  # Will store: {chunk_content: actual_chunk_object}
        chunk_id_map = {}
        chunk_counter = 1

        # Go through each retrieval result
        for query_idx, chunks in enumerate(chunk_lists, 1):

            # Go through each chunk in this query's results
            for position, chunk in enumerate(chunks, 1):  # position is 1-indexed
                #  just pulls out that text string — used here as a unique key to identify the chunk
                #  (since two different documents are unlikely to have the exact same text).

                chunk_content = chunk.page_content

                # Assign a simple ID if we haven't seen this chunk before
                if chunk_content not in chunk_id_map:
                    chunk_id_map[chunk_content] = f"Chunk_{chunk_counter}"
                    chunk_counter += 1

                all_unique_chunks[chunk_content] = chunk # Store the chunk object (in case we haven't seen it before)

                position_score = 1 / (k + position) # Calculate position score: 1/(k + position)

                rrf_scores[chunk_content] += position_score
                # Add position score accrding to page content
                # rrf_scores[doc 2] += 0.016129 <-example value
                # if doc 2 appears again += 0.015873 <-example value
                # then doc 2s total score is 0.032002

        # Sort chunks by RRF score (highest first)
        sorted_chunks = sorted(
            [(all_unique_chunks[chunk_content], score) for chunk_content, score in rrf_scores.items()],
            key=lambda x: x[1],  # Sort by RRF score
            reverse=True  # Highest scores first
        )


        return sorted_chunks

    fused_results = reciprocal_rank_fusion(all_retrieval_results, k=60)

    return fused_results