import os
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import  HuggingFaceEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv

load_dotenv()


def load_documents(docs_path):

    print(f"Loading documents from {docs_path}...")

    if not os.path.exists(docs_path):
        raise FileNotFoundError(
            f"The directory {docs_path} does not exist. Please create it and add your company files.")

    # Load all .txt files from the docs directory
    loader = DirectoryLoader(
        path=docs_path,
        glob="*.txt",
        loader_cls=TextLoader
    )

    documents = loader.load()

    if len(documents) == 0:
        raise FileNotFoundError(f"No .txt files found in {docs_path}. Please add your company documents.")

    for i, doc in enumerate(documents[:len(documents)]):
        print(f"\nDocument {i + 1}:")
        print(f"  Source: {doc.metadata['source']}")
        print(f"  Content length: {len(doc.page_content)} characters")
        print(f"  Content preview: {doc.page_content[:100]}...")
        print(f"  metadata: {doc.metadata}")

    return documents


def split_documents(documents, chunk_size=1000, chunk_overlap=0):
    """Split documents into smaller chunks with overlap"""
    print("Splitting documents into chunks...")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )

    chunks = text_splitter.split_documents(documents)

    if chunks:

        for i, chunk in enumerate(chunks[:5]):
            print(f"\n--- Chunk {i + 1} ---")
            print(f"Source: {chunk.metadata['source']}")
            print(f"Length: {len(chunk.page_content)} characters")
            print(f"Content:")
            print(chunk.page_content)
            print("-" * 50)

        if len(chunks) > 5:
            print(f"\n... and {len(chunks) - 5} more chunks")

    return chunks

def create_vector_store(chunks,persist_directory):
    print("Creating embeddings and storing in ChromaDB")

    if not os.path.exists(persist_directory):
        print("Initializing vector store...\n")

    embedding_model=HuggingFaceEmbeddings(model="all-MiniLM-L6-v2")

    print("--- Creating vector store ---")

    vector_store=Chroma.from_documents(
        documents=chunks,
        persist_directory=persist_directory,
        embedding=embedding_model,
        collection_metadata={"hnsw:space": "cosine"}
    )

    return vector_store



def main():

    docs_path="docs"

    persist_directory="db/chroma_db"

    # Load documents
    documents = load_documents(docs_path)

    # Split into chunks
    chunks = split_documents(documents)

    # Store embeddings in chromaDB
    vector_store=create_vector_store(chunks,persist_directory)



if __name__ == "__main__":
    main()