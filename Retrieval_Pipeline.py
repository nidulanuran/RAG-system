from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
import multi_query_retrieval

load_dotenv()

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

    fused_results=multi_query_retrieval.multi_query_generation(user_question)


    combined_input=f"""Based on the following documents, please answer this question:{user_question}
    Documents:"""

    for i, (doc, score) in enumerate(fused_results, 1):
        combined_input += f"\nDocument {i}:\n{doc.page_content}\n"

    combined_input+=f"""Please provide a clear, helpful answer using only the information from these documents. If you can't find the answer in the documents, say "I don't have enough information to answer that question based on the provided documents."""
    messages=[
        SystemMessage(content="You are a helpful assistant."),
        HumanMessage(content=combined_input)
    ]

    result=model.invoke(messages)
    answer=result.content

    chat_history.append(HumanMessage(content=user_question))
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
