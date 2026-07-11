from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
import Process_pipeline

load_dotenv()

model = ChatGroq(model="openai/gpt-oss-20b")

chat_history=[]
MAX_HISTORY_MESSAGES = 10

def ask_questions(query):

    if chat_history:
        # Tells model to give the new question appropriately,(ex:-"what is the networth of it"=>what is the networth of meta")
        messages=[
            SystemMessage(content="Given the chat history, rewrite the new question to be standalone and searchable. Just return the rewritten question only."),
        ] + chat_history + [
            HumanMessage(content=f"New question :{query}")
        ]

        try:
            result = model.invoke(messages)
            user_question = result.content.strip()
        except Exception as e:
            print(f"[warn] question rewriting failed, using original question: {e}")
            user_question=query

    else:
        user_question=query

    results=Process_pipeline.chunks_reranker(user_question)

    if not results:
        answer = "I don't have enough information to answer that question based on the provided documents."
        print("\n--- Generated Response ---")
        print("Content only:")
        print(answer)
        return answer


    combined_input=f"""Based on the following documents, please answer this question:{user_question}
    Documents:"""

    for i, doc in enumerate(results, 1):
        combined_input += f"\nDocument {i}:\n{doc.page_content}\n"

    combined_input+=f"""Please provide a clear, helpful answer using only the information from these documents. If you can't find the answer in the documents, say "I don't have enough information to answer that question based on the provided documents."""
    messages=[
        SystemMessage(content="You are a helpful assistant."),
        HumanMessage(content=combined_input)
    ]

    try:
        result = model.invoke(messages)
        answer = result.content
    except Exception as e:
        print(f"[warn] final answer generation failed: {e}")
        answer = "Sorry, something went wrong while generating the answer. Please try again."

    chat_history.append(HumanMessage(content=user_question))
    chat_history.append(AIMessage(content=answer))

    del chat_history[:-MAX_HISTORY_MESSAGES]

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
