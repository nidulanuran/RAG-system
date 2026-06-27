from chromadb.segment.impl.vector.hnsw_params import persistent_param_validators
from langchain_huggingface import  HuggingFaceEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv
from torch.nn.functional import embedding

load_dotenv()

embedding_model=HuggingFaceEmbeddings(model="all-MiniLM-L6-v2")

persistent_directory="db/chroma_db"

db=Chroma(
    persist_directory=persistent_directory,
    embedding_function=embedding_model,
    collection_metadata={"hnsw:space": "cosine"}
)

query = "How much did Microsoft pay to acquire GitHub?"

# This will return highest top 3 similarity chunks
retriever=db.as_retriever(search_kwargs={"k":3})

retriever_result=retriever.invoke(query)

print(f"User query :{query}")

for i,doc in enumerate(retriever_result,1):
    print(f"Document {i}:\n{doc.page_content}\n")