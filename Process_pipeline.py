from langchain_huggingface import  HuggingFaceEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from pydantic import BaseModel
from typing import List
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from langchain_core.documents import Document
from langchain_classic.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder

load_dotenv()

# This defines the expected shape of the LLM's output: a JSON object with one key, queries,
# containing a list of strings. Example valid instance:
# {"queries": ["What is RAG?", "Explain retrieval augmented generation", "How does RAG work in LLMs?"]}
class QueryVariations(BaseModel):
    queries: List[str]

embedding_model = HuggingFaceEmbeddings(model="all-MiniLM-L6-v2")

persist_directory_vectors = "db/chroma_db_vectors"

persist_directory_langchain_docs="db/chroma_db_langchain_docs"

db_vectors = Chroma(
    persist_directory=persist_directory_vectors,
    embedding_function=embedding_model,
    collection_metadata={"hnsw:space": "cosine"})

db_langchain_docs=Chroma(
    persist_directory=persist_directory_langchain_docs)

llm = ChatGroq(model="openai/gpt-oss-20b")

# FIX: Forced json_mode to bypass Groq API 400 validation failures.
# TODO: If migrating to ChatOpenAI (e.g., gpt-4o-mini), 'method="json_mode"'
# can be safely removed as OpenAI handles structured outputs natively.
llm_with_tools = llm.with_structured_output(QueryVariations,method="json_mode")


# Chroma's valid include key is "metadatas" (plural), not "metadata".
raw=db_langchain_docs.get(include=["documents","metadatas"])

langchain_docs=[
    Document(page_content=content,metadata=meta)
    for content,meta in zip(raw["documents"],raw["metadatas"])
]

bm25_retriever=BM25Retriever.from_documents(langchain_docs)
bm25_retriever.k=3

vector_retriever = db_vectors.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={"k": 3,
                   "score_threshold": 0.3})

hybrid_retriever = EnsembleRetriever(
    retrievers=[vector_retriever, bm25_retriever],
    weights=[0.7, 0.3]
)

cross_encoder=HuggingFaceCrossEncoder(model_name="BAAI/bge-reranker-base")
reranker=CrossEncoderReranker(model=cross_encoder,top_n=3)


def generate_query_variations(query:str)->List[str]:


    prompt = f"""You are a helpful assistant that generates alternative search queries.
    Generate exactly 3 different variations of the original query that would help retrieve relevant documents from a vector database.
    
    Original query: {query}
    
    You must format your output strictly as a valid JSON object matching the requested schema. Do not add any plain conversational text outside the JSON structure.
    {{"queries": ["variation 1", "variation 2", "variation 3"]}}
"""
    #Must mention that follow the JSON structure

    try:
        response = llm_with_tools.invoke(prompt)
        if response.queries:
            return response.queries
    except Exception as e:
        print(f"[warn] query variation generation failed, falling back to original query: {e}")

    return [query]

def hybrid_search(query_variations:List[str])->List[Document]:

    filtered_chunks:List[Document]=[]
    seen_content=set()

    for question in query_variations:
        try:
            retrieved_chunks = hybrid_retriever.invoke(question)
        except Exception as e:
            print(f"[warn] retrieval failed for variation '{question}': {e}")
            continue

        for doc in retrieved_chunks:
            if doc.page_content not in seen_content:
                seen_content.add(doc.page_content)
                filtered_chunks.append(doc)


    if not filtered_chunks:
        return []

    return filtered_chunks

def chunks_reranker(query:str)->List[Document]:

    query_variations=generate_query_variations(query)
    filtered_chunks=hybrid_search(query_variations)

    try:
        reranked_chunks=reranker.compress_documents(
            documents=filtered_chunks,
            query=query
        )

    except Exception as e:
        print(f"[warn] reranking failed, falling back to unranked candidates: {e}")
        reranked_chunks = filtered_chunks[:3]



    return list(reranked_chunks)
