from collections import defaultdict

from langchain_huggingface import  HuggingFaceEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from pydantic import BaseModel
from typing import List

load_dotenv()

# Pydantic model for structured output
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

    retriever = db.as_retriever(search_kwargs={"k": 5})  # Get more docs for better RRF
    all_retrieval_results = []  # Store all results for RRF

    for i, query in enumerate(query_variations, 1):

        docs = retriever.invoke(query)
        all_retrieval_results.append(docs)  # Store for RRF calculation


    def reciprocal_rank_fusion(chunk_lists, k=60):


        # Data structures for RRF calculation
        rrf_scores = defaultdict(float)  # Will store: {chunk_content: rrf_score}
        all_unique_chunks = {}  # Will store: {chunk_content: actual_chunk_object}

        # For verbose output - track chunk IDs
        chunk_id_map = {}
        chunk_counter = 1

        # Go through each retrieval result
        for query_idx, chunks in enumerate(chunk_lists, 1):


            # Go through each chunk in this query's results
            for position, chunk in enumerate(chunks, 1):  # position is 1-indexed
                # Use chunk content as unique identifier
                chunk_content = chunk.page_content

                # Assign a simple ID if we haven't seen this chunk before
                if chunk_content not in chunk_id_map:
                    chunk_id_map[chunk_content] = f"Chunk_{chunk_counter}"
                    chunk_counter += 1

                chunk_id = chunk_id_map[chunk_content]

                # Store the chunk object (in case we haven't seen it before)
                all_unique_chunks[chunk_content] = chunk

                # Calculate position score: 1/(k + position)
                position_score = 1 / (k + position)

                # Add to RRF score
                rrf_scores[chunk_content] += position_score

        # Sort chunks by RRF score (highest first)
        sorted_chunks = sorted(
            [(all_unique_chunks[chunk_content], score) for chunk_content, score in rrf_scores.items()],
            key=lambda x: x[1],  # Sort by RRF score
            reverse=True  # Highest scores first
        )


        return sorted_chunks

    fused_results = reciprocal_rank_fusion(all_retrieval_results, k=60)

    return fused_results