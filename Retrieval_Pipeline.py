from chromadb.segment.impl.vector.hnsw_params import persistent_param_validators
from langchain_classic.chains.question_answering.map_reduce_prompt import messages
from langchain_huggingface import  HuggingFaceEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv
from torch.nn.functional import embedding
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage,HumanMessage

load_dotenv()

embedding_model=HuggingFaceEmbeddings(model="all-MiniLM-L6-v2")

persistent_directory="db/chroma_db"

db=Chroma(
    persist_directory=persistent_directory,
    embedding_function=embedding_model,
    collection_metadata={"hnsw:space": "cosine"}
)

query = "how mush meta worth?"

# This will return highest top 3 similarity chunks
retriever=db.as_retriever(search_kwargs={"k":3})

retriever_result=retriever.invoke(query)

print(f"User query :{query}")

for i,doc in enumerate(retriever_result,1):
    print(f"Document {i}:\n{doc.page_content}\n")


combined_input=f"""Based on the following documents, please answer this question:{query}

Documents:
{chr(10).join([f"-{doc.page_content}" for doc in retriever_result])}

Please provide a clear, helpful answer using only the information from these documents. If you can't find the answer in the documents, say "I don't have enough information to answer that question based on the provided documents."
"""""

model=ChatGroq(model="openai/gpt-oss-20b")

messages=[
    SystemMessage(content="You are a helpful assistant."),
    HumanMessage(content=combined_input)
]

result=model.invoke(messages)

# Display the  result
print("\n--- Generated Response ---")

print("Content only:")
print(result.content)
