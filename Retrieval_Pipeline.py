from langchain_huggingface import  HuggingFaceEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

load_dotenv()

embedding_model = HuggingFaceEmbeddings(model="all-MiniLM-L6-v2")

persistent_directory = "db/chroma_db"

db = Chroma(
        persist_directory=persistent_directory,
        embedding_function=embedding_model,
        collection_metadata={"hnsw:space": "cosine"}
    )

model = ChatGroq(model="openai/gpt-oss-20b")

chat_history=[]

def ask_questions(query):

    print(f"User query :{query}")

    if chat_history:
        # Tells model to give the new question appropriately,(ex:-"what is the networth of it"=>what is the networth of meta")
        messages=[
            SystemMessage(content="Given the chat history, rewrite the new question to be standalone and searchable. Just return the rewritten question."),
        ] + chat_history + [
            HumanMessage(content=f"New question :{query}")
        ]

        result=model.invoke(messages)
        user_question=result.content.strip()
        print(f"User question:{user_question}")

    else:
        user_question=query

    # This will return highest top 3 similarity chunks
    retriever = db.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={
            "k": 3,
            "score_threshold": 0.3
        })

    retriever_result = retriever.invoke(user_question)

    for i,doc in enumerate(retriever_result,1):
        print(f"Document {i}:\n{doc.page_content}\n")


    combined_input=f"""Based on the following documents, please answer this question:{query}
    Documents:
    {chr(10).join([f"-{doc.page_content}" for doc in retriever_result])}
    Please provide a clear, helpful answer using only the information from these documents. If you can't find the answer in the documents, say "I don't have enough information to answer that question based on the provided documents."
    """
    messages=[
        SystemMessage(content="You are a helpful assistant."),
        HumanMessage(content=combined_input)
    ]

    result=model.invoke(messages)
    answer=result.content

    chat_history.append(HumanMessage(content=query))
    chat_history.append(AIMessage(content=answer))

    # Display the  result
    print("\n--- Generated Response ---")

    print("Content only:")
    print(answer)

    return answer

if __name__ == "__main__":

    while True:
        question=input("Ask questions (q to quit):")

        if question.lower() =="q":
            break

        ask_questions(query=question)
